import torch


class TaskVectorAdapted(TaskVector):
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
                pretrained_state_dict = torch.load(pretrained_checkpoint).state_dict()
                finetuned_state_dict = torch.load(finetuned_checkpoint).state_dict()
                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    if operation == 'subtract':
                        self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]
                    elif operation == 'add':
                        self.vector[key] = finetuned_state_dict[key] + pretrained_state_dict[key]
                    elif operation == 'multiply':
                        self.vector[key] = finetuned_state_dict[key] * pretrained_state_dict[key]
                    elif operation == 'divide':
                        self.vector[key] = finetuned_state_dict[key] / pretrained_state_dict[key]
                    else:
                        raise ValueError(f"Unknown operation: {operation}. Supported operations are: 'subtract', 'add', 'multiply', 'divide'.")
   
