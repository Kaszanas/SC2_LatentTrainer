import lightning as pl


class LSupervisedGuidedVAE(pl.LightningModule):
    def __init__(self):
        super().__init__()
