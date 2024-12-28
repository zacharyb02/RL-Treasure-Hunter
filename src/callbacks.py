from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

class SuccessRateCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super(SuccessRateCallback, self).__init__(*args, **kwargs)
        self.success_count = 0
        self.attempts = 0

    def _on_step(self):
        super()._on_step()

        # Accessing dones from self.locals which is provided by stable-baselines3 during the step
        if 'dones' in self.locals:
            done_array = self.locals['dones']
            if done_array.any():  # If any done signal is True
                self.attempts += sum(done_array)  # Count all episodes that ended
                # Count successes based on some condition, e.g., reaching the goal
                for idx, done in enumerate(done_array):
                    if done and self.locals['rewards'][idx] > 50:  # Assume reward > 50 means success
                        self.success_count += 1

        return True

    def get_success_rate(self):
        return self.success_count / self.attempts if self.attempts else 0
