from torchvision import datasets
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as transforms


# データセットの読み込み
ds_train = datasets.FashionMNIST(
    root='dataset',
    train=True,
    download=True
)

print(f'dataset size: {len(ds_train)}')

# インデックスを指定してデータを取り出す
# 画像とクラス番号の組になってる
image, target = ds_train[59999]

print(type(image))
print(target)

# plt.imshow(image, cmap='gray_r', vmin=0, vmax=255)
# plt.title(target)
# plt.show()

fig, ax = plt.subplots()
ax.imshow(image, cmap='gray_r', vmin=0, vmax=255)
ax.set_title(target)
plt.show()

image = transforms.functional.to_image(image)
image = transforms.functional.to_dtype(image, scale=True)
print(type(image))
print(image.shape, image.dtype)
print(image.min(), image.max())
