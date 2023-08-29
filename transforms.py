from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image

# python function -> tensor data type
# use transforms.totensor to solve two problems

# 2.tensor data type advantages

img_path = "test_dataset/train/ants_image/0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")


# 1.how to use transform(python)
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer.add_image("Tensor_img", tensor_img)

writer.close()



