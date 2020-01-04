from torchvision import transforms

def pil2tensor(pil):  
  t = transforms.Compose([
      transforms.Resize(256),
      transforms.ToTensor(),
      transforms.Normalize(         # ImageNet normalization
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
      )
  ])
  return t(pil).to(device, dtype)

def tensor2pil(img):
  # Reverse ImageNet normalization
  t = transforms.Compose([
    transforms.Normalize(
      mean=[0, 0, 0],
      std=[1/0.229, 1/0.224, 1/0.225]
    ),
    transforms.Normalize(
      mean=[-0.485, -0.456, -0.406],
      std=[1, 1, 1]
    )
  ])
  return t(img).clamp_(0,1)
