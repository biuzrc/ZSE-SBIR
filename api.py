from .model.model import Model
from .utils.util import load_checkpoint
from collections import namedtuple

def load_ZES_SBIR_model(batch=2):
    config =  {
        'cls_number': 100,
        'd_model': 768,
        'd_ff': 1024,
        'head': 8,
        'number': 1,
        'pretrained': True,
        'anchor_number': 49,
        'batch': batch
    }
    Config = namedtuple('Config', config.keys())
    config = Config(**config)


    model = Model(config)
    checkpoint = load_checkpoint("/home/Sketch-SG-Image/ZSE_SBIR/checkpoints/best_checkpoint.pth")

    cur = model.state_dict()
    new = {k: v for k, v in checkpoint['model'].items() if k in cur.keys()}
    cur.update(new)
    model.load_state_dict(cur)

    return model    




if __name__ == '__main__':
    model = load_ZES_SBIR_model()
    print(model)