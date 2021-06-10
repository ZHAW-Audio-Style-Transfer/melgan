from models.MelGAN import MelGAN

ModelsMap = {
    "MelGAN": MelGAN,
}


def getModel(name: str):
    return ModelsMap[name]()
