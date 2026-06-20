import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components

class ICAProcessor:
    def __init__(self, n_components=0.99, random_state=97, max_iter='auto', method="infomax"):
        fit_params = dict(extended=True) if method == "infomax" else None
        
        self.ica = ICA(
            n_components=n_components, 
            random_state=random_state, 
            max_iter=max_iter, 
            method=method, 
            fit_params=fit_params
        )
        self.fitted = False

    def fit(self, epoch):
        self.ica.fit(epoch)
        self.epoch = epoch
        self.fitted = True

    def plot_components(self):
        if not self.fitted:
            raise RuntimeError("ICA must be fitted before plotting components.")
        self.ica.plot_sources(self.epoch, show_scrollbars=True)
        self.ica.plot_components(inst=self.epoch)
        self.ica.plot_properties(self.epoch, verbose=False)
    
    def __ica_labels(self, method="iclabel"):
        if not self.fitted:
            raise RuntimeError("ICA must be fitted before labeling components.")

        return label_components(self.epoch, self.ica, method=method)
    
    def get_exclude_components(self, method="iclabel", threshold=0.80):

        labels_dict = self.__ica_labels(method=method)
        labels = labels_dict["labels"]
        probs = labels_dict["y_pred_proba"]
        
        exclude = []
        print(f"Avaliable componentes found: {labels}")
        for i, (label, prob) in enumerate(zip(labels, probs)):
            if label in ['eye blink', 'muscle artifact'] and prob >= threshold:
                exclude.append(i)
                print(f"-> Componente {i} excluída: {label} (Probabilidade: {prob:.2f})")
            else:
                pass 
                
        evoked = self.epoch.average()
        self.ica.plot_overlay(evoked, exclude=exclude, picks="eeg")
        
        if exclude: 
            self.ica.plot_properties(self.epoch, picks=exclude, verbose=False)
            
        return exclude
    
    def apply_ica(self, exclude):
        if not self.fitted:
            raise RuntimeError("ICA must be fitted before applying.")
        self.ica.exclude = exclude
        return self.ica.apply(self.epoch.copy())