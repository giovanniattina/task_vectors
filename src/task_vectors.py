import torch
from artificial_data.model_art import BackBone

class TaskVector():
    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None, operation='subtract'):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.

        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        if vector is not None:
            self.vector = vector
        else:
            assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
            with torch.no_grad():
                pretrained_state_dict = torch.load(pretrained_checkpoint)['state_dict']
                finetuned_state_dict = torch.load(finetuned_checkpoint)['state_dict']
                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8] or 'head' in key:
                        continue
                    if operation == 'subtract':
                        self.vector[key] = finetuned_state_dict[key] - \
                            pretrained_state_dict[key]
                    elif operation == 'add':
                        self.vector[key] = finetuned_state_dict[key] + \
                            pretrained_state_dict[key]
                    elif operation == 'multiply':
                        self.vector[key] = finetuned_state_dict[key] * \
                            pretrained_state_dict[key]
                    elif operation == 'divide':
                        self.vector[key] = finetuned_state_dict[key] / \
                            pretrained_state_dict[key]
                    else:
                        raise ValueError(
                            f"Unknown operation: {operation}. Supported operations are: 'subtract', 'add', 'multiply', 'divide'.")

    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(
                        f'Warning, key {key} is not present in both task vectors.')
                    continue
                elif 'head' in key:
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def sub_interpolate(self, other, alpha=0.5):
        """Interpolate between two task vectors."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(
                        f'Warning, key {key} is not present in both task vectors.')
                    continue
                elif 'head' in key:
                    continue
                new_vector[key] = (1 - alpha) * self.vector[key] + alpha * other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = - self.vector[key]
        return TaskVector(vector=new_vector)

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            checkpoint = torch.load(pretrained_checkpoint)
            original_input_dim = checkpoint['model_config'].get('input_dim', 20)  # Default based on dataset1

            pretrained_model = BackBone(original_input_dim)
            pretrained_model.load_state_dict(checkpoint['state_dict'])


            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(
                        f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                    continue
                elif 'head' in key:
                    continue
                new_state_dict[key] = pretrained_state_dict[key] + \
                    scaling_coef * self.vector[key]
        pretrained_model.load_state_dict(new_state_dict, strict=False)
        return pretrained_model
