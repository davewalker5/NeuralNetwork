"""Interactive Matplotlib wrapper around the existing trainable neuron."""

from math import isfinite
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .boundary_experiment import ExperimentSettings, boundary, state_at
from .trainable_neuron import TrainableSingleInputNeuron, train


class BoundaryExplorer:
    """Keep pending controls separate from the settings of the displayed run."""

    def __init__(self) -> None:
        """Create a desktop figure, controls and the default deterministic run."""
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, CheckButtons, Slider

        # Axes positions are fractions of the figure: left, bottom, width, height.
        # Reserve space between the plots and sliders for diagnostics and readiness.
        self.figure = plt.figure(figsize=(11, 7))
        self.figure.suptitle("Trainable neuron: explore the gap between 2 and 4")
        self.status = self.figure.text(
            0.07, 0.405, "Starting…", fontsize=11, weight="bold"
        )
        self.probability = self.figure.add_axes((0.07, 0.61, 0.58, 0.31))
        self.history = self.figure.add_axes((0.73, 0.75, 0.24, 0.17))
        self.loss = self.figure.add_axes((0.73, 0.45, 0.24, 0.19))
        self.summary = self.figure.text(0.07, 0.46, "", fontsize=10)
        self.pending = self.figure.text(0.07, 0.015, "", fontsize=10)
        # Each specification supplies the slider's limits, default and step size.
        # Only the update count needs integer steps; the other values are continuous.
        self.controls = {}
        specs = (
            ("centre", "Requested boundary", 2.02, 3.98, 3.0, None),
            ("spacing", "Pair half-spacing", 0.01, 0.5, 0.1, None),
            ("learning_rate", "Learning rate", 0.001, 1.0, 0.1, None),
            ("epochs", "Training updates", 1, 20000, 2000, 1),
        )
        for index, (key, label, low, high, value, step) in enumerate(specs):
            axis = self.figure.add_axes((0.23, 0.36 - index * 0.065, 0.40, 0.025))
            slider = Slider(
                axis,
                label,
                low,
                high,
                valinit=value,
                valstep=step,
                valfmt="%d" if key == "epochs" else "%.3f",
            )
            self.controls[key] = slider
            slider.on_changed(self.settings_changed)
        self.include = CheckButtons(
            self.figure.add_axes((0.72, 0.28, 0.25, 0.08)),
            ["Add generated pair"],
            [True],
        )
        self.include.on_clicked(self.settings_changed)
        self.train_button = Button(
            self.figure.add_axes((0.74, 0.19, 0.10, 0.05)), "Train"
        )
        self.reset_button = Button(
            self.figure.add_axes((0.86, 0.19, 0.10, 0.05)), "Reset"
        )
        self.train_button.on_clicked(self.run)
        self.reset_button.on_clicked(self.reset)
        self.replay = Slider(
            self.figure.add_axes((0.23, 0.08, 0.70, 0.03)),
            "Replay epoch",
            0,
            2000,
            valinit=2000,
            valstep=1,
            valfmt="%d",
        )
        self.replay.on_changed(self.render)
        # Proposed controls and the completed run are kept separately so moving a
        # slider cannot silently change the dataset described by the current plots.
        self.adjusting = False
        self.result = None
        self.settings = None
        self.busy = False
        self.executor = ThreadPoolExecutor(max_workers=1)
        # The worker only calculates training results. A GUI timer checks for
        # completion so all Matplotlib changes happen on the main thread.
        self.completion_timer = self.figure.canvas.new_timer(interval=100)
        self.completion_timer.add_callback(self.finish_training)
        self.figure.canvas.mpl_connect("close_event", self.close)
        self.run()

    def close(self, _event: object = None) -> None:
        """Stop completion polling and release the worker when the window closes."""
        self.completion_timer.stop()
        # Avoid waiting in the close handler. An already-running task finishes;
        # queued tasks are cancelled and no timer remains to update the closed UI.
        self.executor.shutdown(wait=False, cancel_futures=True)

    def set_busy(self, busy: bool) -> None:
        """Disable experiment controls while a background run is being prepared.

        :param busy: Whether training or rendering the new run is in progress.
        """
        self.busy = busy
        widgets = [
            *self.controls.values(),
            self.include,
            self.train_button,
            self.reset_button,
            self.replay,
        ]
        for widget in widgets:
            # The active property disables interaction without toggling the
            # checkbox's selected state; alpha gives matching visual feedback.
            widget.active = not busy
            widget.ax.set_alpha(0.45 if busy else 1.0)

    def read_settings(self) -> ExperimentSettings:
        """Read the proposed dataset and training configuration from controls."""
        # Build a validated snapshot rather than passing mutable widgets to training.
        return ExperimentSettings(
            centre=float(self.controls["centre"].val),
            spacing=float(self.controls["spacing"].val),
            augmented=self.include.get_status()[0],
            learning_rate=float(self.controls["learning_rate"].val),
            epochs=int(self.controls["epochs"].val),
        )

    def settings_changed(self, _value: object = None) -> None:
        """Clamp the centre for the current spacing and label unapplied changes."""
        if self.adjusting or self.busy:
            return
        spacing = self.controls["spacing"].val
        centre = self.controls["centre"]
        # Leave a 0.01 margin inside each original endpoint, even when spacing grows.
        clamped = float(np.clip(centre.val, 2 + spacing + 0.01, 4 - spacing - 0.01))
        # set_val invokes slider callbacks too; suppress that nested callback while
        # applying the corrected centre to avoid calling this method recursively.
        self.adjusting = True
        centre.set_val(clamped)
        self.adjusting = False
        # Updating this message proposes a new experiment without replacing the run.
        proposed = self.read_settings()
        pair = f"({proposed.centre - spacing:.3f}, 0), ({proposed.centre + spacing:.3f}, 1)"
        status = (
            "Pending changes — press Train"
            if proposed != self.settings
            else "Settings match displayed run"
        )
        self.pending.set_text(
            f"{status}. Proposed pair: {pair}"
            if proposed.augmented
            else f"{status}. Proposed dataset: originals only"
        )
        self.figure.canvas.draw_idle()

    def run(self, _event: object = None) -> None:
        """Train in a worker so the window can paint its status and stay responsive."""
        if self.busy:
            return
        proposed = self.read_settings()
        # Disable controls before submitting work so another click cannot enqueue
        # a second run or alter the settings associated with the pending result.
        self.set_busy(True)
        self.status.set_text(f"Training {proposed.epochs:,} updates… please wait")
        self.status.set_color("darkorange")
        self.pending.set_text("Controls will become available when training finishes.")
        self.figure.canvas.draw_idle()
        self.proposed = proposed
        # Every experiment starts from zero parameters for comparable results.
        # Training runs off the GUI thread so the window can paint the busy message.
        self.future = self.executor.submit(
            train,
            TrainableSingleInputNeuron(),
            *proposed.dataset(),
            learning_rate=proposed.learning_rate,
            epochs=proposed.epochs,
        )
        self.completion_timer.start()

    def finish_training(self) -> None:
        """Apply a completed worker result on the GUI thread and announce readiness."""
        if not self.busy or not self.future.done():
            return
        # Only read the future once it is done; otherwise result() would block the UI.
        self.completion_timer.stop()
        proposed = self.proposed
        try:
            result = self.future.result()
        except Exception as error:
            # Worker exceptions surface here. Preserve the last successful run and
            # restore controls so the learner can adjust settings and retry.
            self.set_busy(False)
            self.status.set_text("Training failed — controls available to retry")
            self.status.set_color("tab:red")
            self.pending.set_text(
                f"Training failed: {error}. "
                + (
                    "Previous run remains displayed."
                    if self.result is not None
                    else "Change settings and press Train to retry."
                )
            )
            self.figure.canvas.draw_idle()
            return
        self.result = result
        self.settings = proposed
        # Step evaluations describe the state BEFORE each update. Appending the
        # final state gives histories indexed by completed updates, from 0 to N.
        self.boundaries = np.array(
            [boundary(step.before) for step in result.steps] + [boundary(result.neuron)]
        )
        self.losses = np.array(
            [step.evaluation.loss for step in result.steps]
            + [result.final_evaluation.loss]
        )
        self.replay.valmax = proposed.epochs
        self.replay.ax.set_xlim(0, proposed.epochs)
        # Selecting the final epoch invokes render() with the new trace. Announce
        # readiness only after its plots and diagnostics have been prepared.
        self.replay.set_val(proposed.epochs)
        self.set_busy(False)
        self.status.set_text("Ready — explore the controls or replay epochs")
        self.status.set_color("darkgreen")
        self.settings_changed()

    def reset(self, _event: object = None) -> None:
        """Restore the default configuration and its corresponding trained run."""
        if self.busy:
            return
        # Reset all widgets as one operation rather than clamping the centre against
        # a mixture of old and default values as individual callbacks fire.
        self.adjusting = True
        for slider in self.controls.values():
            slider.reset()
        if not self.include.get_status()[0]:
            self.include.set_active(0)
        self.adjusting = False
        self.run()

    def render(self, _value: object = None) -> None:
        """Draw all views from a single recorded epoch without training again."""
        if self.result is None:
            return
        epoch = int(self.replay.val)
        # Select matching parameters and loss once, then use them in every view.
        # Replay inspects the existing trace and never calls the training function.
        neuron, evaluation = state_at(self.result, epoch)
        threshold = boundary(neuron)
        inputs, targets = self.settings.dataset()
        # These nearest opposing examples bound a correctly classifying increasing
        # sigmoid: the lower endpoint allows a class-zero tie, the upper does not.
        lower = max(x for x, y in zip(inputs, targets) if y == 0)
        upper = min(x for x, y in zip(inputs, targets) if y == 1)
        x = np.linspace(-0.3, 6.3, 700)
        probabilities = np.array([neuron.forward(float(value)) for value in x])
        axis = self.probability
        # Rebuild the selected-epoch view so old markers and boundary lines cannot
        # survive a dataset change. Fixed limits keep probability curves comparable.
        axis.clear()
        axis.axvspan(
            lower,
            upper,
            color="green",
            alpha=0.12,
            label="Data interval [lower, upper)",
        )
        axis.plot(x, probabilities, label="Probability", color="tab:blue")
        axis.step(
            x,
            (probabilities > 0.5).astype(int),
            where="mid",
            alpha=0.4,
            color="tab:purple",
            label="On/off classification",
        )
        axis.scatter(
            inputs[:6], targets[:6], label="Original examples", color="black", zorder=5
        )
        if self.settings.augmented:
            axis.scatter(
                inputs[6:],
                targets[6:],
                marker="D",
                s=65,
                color="tab:orange",
                label="Added examples",
                zorder=6,
            )
        axis.axhline(0.5, color="grey", linestyle=":")
        axis.axvline(
            self.settings.centre,
            color="tab:orange",
            linestyle="--",
            label="Requested centre",
        )
        if isfinite(threshold):
            # Undefined boundaries (including the initial zero weight) have no line.
            axis.axvline(threshold, color="tab:red", label="Learned boundary")
        axis.set(
            xlim=(-0.3, 6.3),
            ylim=(-0.08, 1.08),
            xlabel="Input",
            ylabel="Probability / class",
            title="Selected epoch: probability and on/off point",
        )
        axis.legend(loc="upper left", fontsize=8)
        # History views retain the entire run; replay moves only their selection
        # markers. NaN boundaries naturally leave gaps rather than false zeroes.
        history = self.history
        history.clear()
        history.axhspan(lower, upper, color="green", alpha=0.12)
        history.axhline(self.settings.centre, color="tab:orange", linestyle="--")
        history.plot(self.boundaries, color="tab:red")
        history.axvline(epoch, color="grey", linestyle=":")
        if isfinite(threshold):
            history.plot(epoch, threshold, "o", color="tab:red")
        history.set(xlabel="Epoch", ylabel="Boundary", title="Boundary movement")
        loss = self.loss
        loss.clear()
        loss.plot(self.losses, color="tab:blue")
        loss.axvline(epoch, color="grey", linestyle=":")
        loss.plot(epoch, evaluation.loss, "o", color="tab:blue")
        loss.set(xlabel="Epoch", ylabel="Mean BCE", title="Training loss")
        # Apply the same strict threshold as the demo: exactly 0.5 means class zero.
        correct = sum(int(p > 0.5) == y for p, y in zip(evaluation.outputs, targets))
        boundary_text = (
            f"{threshold:.5f}"
            if isfinite(threshold)
            else "undefined (weight near zero)"
        )
        if isfinite(threshold) and not -0.3 <= threshold <= 6.3:
            # Report excursions numerically even when the probability plot cannot
            # show the red line; the boundary history still includes those values.
            boundary_text += " (outside probability plot)"
        pair = (
            f"Added: ({inputs[6]:.3f}, 0), ({inputs[7]:.3f}, 1)"
            if self.settings.augmented
            else "Original examples only"
        )
        self.summary.set_text(
            f"Displayed run: {pair}; rate={self.settings.learning_rate:.3f}, updates={self.settings.epochs}\n"
            f"Epoch {epoch}: w={neuron.weight:.5f}, b={neuron.bias:.5f}, boundary={boundary_text}\n"
            f"Mean loss={evaluation.loss:.6f}; correct={correct}/{len(inputs)}; data interval=[{lower:.3f}, {upper:.3f})"
        )
        self.figure.canvas.draw_idle()


def main() -> None:
    """Launch a desktop explorer, explaining missing dependencies or backends."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.backends.registry import backend_registry
    except ImportError as error:
        raise SystemExit(
            "Install exploration dependencies: venv/bin/python -m pip install '.[exploration]'"
        ) from error
    try:
        # An image-only backend can draw the figure but cannot run its widgets or
        # completion timer, so reject it before starting a background training run.
        _, framework = backend_registry.resolve_backend(matplotlib.get_backend())
        if framework is None:
            raise RuntimeError("a non-interactive backend is selected")
        explorer = BoundaryExplorer()
        plt.show()
        # Keep widget callbacks alive until the window closes.
        del explorer
    except (ImportError, RuntimeError) as error:
        raise SystemExit(
            f"Cannot open the explorer: {error}. On macOS try MPLBACKEND=MacOSX "
            "venv/bin/python -m src.single_neuron.trainable_neuron_explorer. "
            "A working desktop GUI backend is required."
        ) from error


if __name__ == "__main__":
    main()
