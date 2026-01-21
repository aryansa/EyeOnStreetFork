import torch
import timm

def load_vit_timm(model_path, device):
    """
    Loads a ViT model using timm instead of DINOv2.

    Args:
        model_path (str): Path to saved model weights.
        device (str or torch.device): Device to load the model on.

    Returns:
        model (torch.nn.Module): Vision Transformer model.
    """

    # Create a ViT-S/14 model (closest match to vit_small)
    model = timm.create_model(
        "vit_small_patch14_518",
        pretrained=False,
        num_classes=0  # removes classification head
    )

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()
    return model
