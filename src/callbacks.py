from pytorch_lightning.callbacks import Callback


class EmissionsTrackerCallback(Callback):
    # Lightning callback for tracking training emissions using CodeCarbon

    def __init__(self, log_to_trainer: bool = True):
        super().__init__()
        from codecarbon import EmissionsTracker
        self.tracker = EmissionsTracker(log_level="error", allow_multiple_runs=True)
        self.log_to_trainer = log_to_trainer

    def on_fit_start(self, trainer, pl_module):
        # Start emissions tracking at beginning of training
        self.tracker.start()

    def on_train_epoch_end(self, trainer, pl_module):
        # Flush and log per-epoch emissions
        self.tracker.flush()
        data = self.tracker._prepare_emissions_data()
        emissions_g = data.emissions * 1000
        elapsed_s   = data.duration
        print(f"[EMISSIONS] epoch {trainer.current_epoch}: {emissions_g:.1f} g CO₂ — {elapsed_s:.1f}s")
        if self.log_to_trainer:
            pl_module.log("carbon_emissions", emissions_g, on_epoch=True, prog_bar=False)
            pl_module.log("training_time_s", elapsed_s, on_epoch=True, prog_bar=False)

    def on_fit_end(self, trainer, pl_module):
        # Stop tracking and print total emissions
        self.tracker.stop()
        data = self.tracker._prepare_emissions_data()
        total_g   = data.emissions * 1000
        total_min = data.duration / 60
        print(f"[EMISSIONS TOTAL] {total_g:.1f} g CO₂ — {total_min:.1f} min")
