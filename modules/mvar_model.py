from scot.var import VAR
from scot.varica import mvarica
from scot.connectivity import connectivity
from scot.connectivity_statistics import bootstrap_connectivity as bc, significance_fdr, surrogate_connectivity
from scot.datatools import randomize_phase

import numpy as np
from joblib import Parallel, delayed
import scipy

from utils.validation_model import biggest_eigenvector_mvar, calculate_bic

class mvar_optimized():
    def __init__(self, epochs):
        self.epochs = epochs
        self.data =self.epochs.get_data()
        self.sfreq = self.epochs.info['sfreq']

        self.fitted = False
        self.result_var = None
        self.optimal_order = None
        self.delta = None

        self.__var_base = None
    
    def find_order_delta(self, min_p=7, max_p=25, deltas = [0.001, 0.01, 0.1], top_k=5, n_jobs=-1):

        var_optimizer = VAR(model_order=min_p)

        var_optimizer.optimize_order(self.data, min_p=min_p, max_p=max_p, n_jobs=n_jobs)
        optimal_p = var_optimizer.p

        print(f"Ordem ótima encontrada: p={optimal_p}")
        print(f"Otimizando regularização (delta) via busca binária...")

        var_optimizer.optimize_delta_bisection(self.data)
        optimal_delta = var_optimizer.delta

        print(f"Delta ótimo encontrado: delta={optimal_delta:.5f}")

        return [{"order": optimal_p, "delta": optimal_delta, "bic": None}]
        

        results = []

        for delta in deltas:
            for p in range(min_p, max_p + 1):
                var_test_cv = VAR(model_order=p, delta=delta)
                mvar_reults = mvarica(
                    x=self.data,
                    var=var_test_cv,
                    reducedim='no_pca',
                    optimize_var=False,
                    varfit='ensemble',
                )
                bic = calculate_bic(var_test_cv, mvar_reults, self.data.shape[2], self.data.shape[0])
                print(f"p={p:>2}, delta={delta:.3f}, BIC={bic:.4f}")
                results.append((bic, p, delta))
        
        # Sort results by BIC in ascending order (lower is better)
        results.sort(key=lambda x: x[0])
        top_results = results[:top_k]
        
        print(f"\nTop {len(top_results)} results:")
        for i, (bic, p, delta) in enumerate(top_results, 1):
            print(f"{i}. Order: {p} with delta: {delta} and BIC value: {bic:.4f}")
        
        return [{"order": p, "delta": delta, "bic": bic} for bic, p, delta in top_results]
    
    def fit_model(self, optimal_order, delta):
        self.__var_base = VAR(model_order=optimal_order, delta=delta)
        self.result_var = mvarica(
                    x=self.data,
                    var=self.__var_base,
                    reducedim='no_pca',
                    optimize_var=False,
                    varfit='ensemble',
                )
        self.fitted = True
        self.optimal_order = optimal_order 
        self.delta = delta
        print("Model fitted with optimal order and delta.")
        self.__validate_model()
    
    def __validate_model(self):
        if not self.fitted:
            raise ValueError("Model must be fitted before validation.")
        
        var_source = self.result_var.b
        if not var_source.is_stable():
            print("Attention: The fitted VAR model is not stable. Consider adjusting the order or delta.")
        else:
            print("The fitted VAR model is stable.")

        n_times = self.data.shape[2]
        h = int(np.sqrt(n_times))
        p_val, q0, q = var_source.test_whiteness(h=h, repeats=200, get_q=True)
        print(f"Whiteness Test p-value: {p_val:.4f}")
        if p_val < 0.05:
            print("Warning: Residuals are not white. Consider increasing the model order or adjusting delta.")
        else:
            print("Residuals appear to be white, indicating a good model fit.")
        
        biggest_eigenvector = biggest_eigenvector_mvar(var_source)
        print(f"\nBiggest eigenvector: {biggest_eigenvector:.4f} (Must be < 1 for stability)")
    
    def get_connectivity(self, measure='PDC', nfft=512, freq_min=8.0, freq_max=30.0):
        if not self.fitted:
            raise ValueError("Model must be fitted before computing connectivity.")
        
        var_eeg = self.result_var.a

        connectivity_result = connectivity(
            b=var_eeg.coef,
            c=var_eeg.rescov,
            nfft=nfft,
            measure_names=[measure],
        )
        freqs = np.linspace(0, self.sfreq / 2, nfft)

        idx_banda = np.where((freqs >= freq_min) & (freqs <= freq_max))[0]

        pdc_completo_eeg = connectivity_result['PDC']
        pdc_banda_eeg = pdc_completo_eeg[:, :, idx_banda]

        pdc_matrix_eeg = np.mean(pdc_banda_eeg, axis=2)
        np.fill_diagonal(pdc_matrix_eeg, 0)

        return connectivity_result[measure], pdc_matrix_eeg
    
    def validation_connectivity(self, real_matric, measure='PDC', n_surrogates=300, alpha=0.05, nfft=512, freq_min=8.0, freq_max=30.0, n_jobs=-1):
        if not self.fitted:
            raise ValueError("Model must be fitted before validating connectivity.")
        
        if self.optimal_order is None or self.delta is None:
            raise ValueError("optimal_order or delta is None! Did you forget to save them in fit_model?")
        
        print(f"Performing rigorous surrogate connectivity analysis with {n_surrogates} surrogates using MVARICA...")
        print(f"Parallelizing across {n_jobs if n_jobs != -1 else 'ALL'} CPU cores...")

        # Dispara os cálculos paralelos. 
        # Note que passamos os atributos de 'self' diretamente como parâmetros.
        surrogate_matrices = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(_worker_compute_surrogate)(
                self.data,           # Apenas a matriz numpy
                self.sfreq,          # Frequência de amostragem
                self.optimal_order,  # Ordem do modelo
                self.delta,          # Delta da regularização
                measure, nfft, freq_min, freq_max
            )
            for _ in range(n_surrogates)
        )

        matric_surrogates = np.array(surrogate_matrices)
        
        p_values = np.mean(matric_surrogates >= real_matric, axis=0)

        s_fdr = significance_fdr(p_values, alpha)
        real_validated_matric = real_matric.copy()
        real_validated_matric[~s_fdr] = 0.0 

        return real_validated_matric
    
    def bootstrap_connectivity(self, measure='PDC', n_bootstraps=300, nfft=512):
        if not self.fitted:
            raise ValueError("Model must be fitted before performing bootstrap analysis.")

        print(f"Performing bootstrap connectivity analysis with {n_bootstraps} bootstraps...")
        bootstrap_results = bc(
            measures=[measure], 
            data=self.data,      
            var=self.__var_base,
            nfft=nfft, 
            repeats=n_bootstraps,
        )

        # Calcula os limites (2.5% e 97.5% criam um intervalo de 95%)
        ci_lower = np.percentile(bootstrap_results[measure], 2.5, axis=0)
        ci_upper = np.percentile(bootstrap_results[measure], 97.5, axis=0)

        return bootstrap_results[measure], ci_lower, ci_upper


def _worker_compute_surrogate(data, sfreq, optimal_order, delta, measure, nfft, freq_min, freq_max):
    scipy.shape = np.shape
    scipy.cov = np.cov
    scipy.zeros = np.zeros
    scipy.ceil = np.ceil
    scipy.atleast_3d = np.atleast_3d    
    scipy.eye = np.eye
    scipy.sum = np.sum
    scipy.sqrt = np.sqrt
    scipy.exp = np.exp
    scipy.sign = np.sign

    # Embaralhar fases
    shuffled_data = randomize_phase(data)
    
    # Criamos o motor VAR aqui dentro para cada núcleo ter o seu limpo!
    var_base = VAR(model_order=optimal_order, delta=delta)
    
    # Ajustar o MVARICA
    res_var_surrogate = mvarica(
        x=shuffled_data,
        var=var_base,
        reducedim='no_pca',
        optimize_var=False,
        varfit='ensemble',
    )
    
    # Extrair a conectividade do EEG
    var_eeg_falso = res_var_surrogate.a
    conn_falsa = connectivity(
        b=var_eeg_falso.coef,
        c=var_eeg_falso.rescov,
        nfft=nfft,
        measure_names=[measure]
    )
    
    # Processamento da banda de frequência
    freqs = np.linspace(0, sfreq / 2, nfft)
    idx_banda = np.where((freqs >= freq_min) & (freqs <= freq_max))[0]
    
    pdc_falso_banda = conn_falsa[measure][:, :, idx_banda]
    pdc_falso_matrix = np.mean(pdc_falso_banda, axis=2)
    np.fill_diagonal(pdc_falso_matrix, 0)
    
    return np.abs(pdc_falso_matrix)