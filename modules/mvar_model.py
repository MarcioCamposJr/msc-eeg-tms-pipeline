from scot.var import VAR
from scot.varica import mvarica
from scot.connectivity import connectivity
from scot.connectivity_statistics import surrogate_connectivity, significance_fdr

import numpy as np

from utils.validation_model import biggest_eigenvector_mvar, calculate_bic

class mvar_optimized():
    def __init__(self, epochs):
        self.epochs = epochs
        self.data =self.epochs.get_data()
        self.sfreq = self.epochs.info['sfreq']

        self.fitted = False
        self.result_var = None
        self.optimal_order = None
    
    def find_order_delta(self, min_p=7, max_p=25, deltas = [0.001, 0.01, 0.1], top_k=5):
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
        var_base = VAR(model_order=optimal_order, delta=delta)
        self.result_var = mvarica(
                    x=self.data,
                    var=var_base,
                    reducedim='no_pca',
                    optimize_var=False,
                    varfit='ensemble',
                )
        self.fitted = True
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
    
    def validation_connectivity(self, real_matric, measure='PDC', n_surrogates=300, alpha=0.05, nfft=512):
        if not self.fitted:
            raise ValueError("Model must be fitted before validating connectivity.")
        
        var_eeg = self.result_var.a

        print(f"Performing surrogate connectivity analysis with {n_surrogates} surrogates...")
        surrogate_results = surrogate_connectivity(
            measure_names=[measure], 
            data=self.data,      
            var=var_eeg,
            nfft=nfft, 
            repeats=n_surrogates
        )

        matric_surrogates = np.abs(surrogate_results[measure])
        p_values = np.mean(matric_surrogates >= real_matric, axis=0)

        s_fdr = significance_fdr(p_values, alpha)
        real_validated_matric = real_matric.copy()
        real_validated_matric[~s_fdr] = 0.0 

        return real_validated_matric
    
    def bootstrap_connectivity(self, measure='PDC', n_bootstraps=300, nfft=512):
        if not self.fitted:
            raise ValueError("Model must be fitted before performing bootstrap analysis.")
        
        var_eeg = self.result_var.a

        print(f"Performing bootstrap connectivity analysis with {n_bootstraps} bootstraps...")
        bootstrap_results = surrogate_connectivity(
            measure_names=[measure], 
            data=self.data,      
            var=var_eeg,
            nfft=nfft, 
            repeats=n_bootstraps,
            method='bootstrap'
        )

        # Calcula os limites (2.5% e 97.5% criam um intervalo de 95%)
        ci_lower = np.percentile(bootstrap_results[measure], 2.5, axis=0)
        ci_upper = np.percentile(bootstrap_results[measure], 97.5, axis=0)

        return bootstrap_results[measure], ci_lower, ci_upper