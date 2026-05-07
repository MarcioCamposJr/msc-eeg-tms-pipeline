from mne.preprocessing import ICA
from mne_icalabel import label_components

class ICAProcessor:
    def __init__(self, random_state=97, max_iter=800, method="infomax"):
        self.ica = ICA(random_state=random_state, max_iter=max_iter, method=method)
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
    
    def get_exclude_components(self, method="iclabel", threshold=0.5):
        labels = self.__ica_labels(method=method)
        exclude = [i for i, prob in enumerate(labels["y_pred_proba"]) if prob > threshold]
        evoked = self.epoch.average()
        self.ica.plot_overlay(evoked, exclude=exclude, picks="eeg")
        self.ica.plot_properties(self.epoch, picks=exclude, verbose=False)
        return exclude
    
    def apply_ica(self, exclude):
        if not self.fitted:
            raise RuntimeError("ICA must be fitted before applying.")
        self.ica.exclude = exclude
        return self.ica.apply(self.epoch.copy())