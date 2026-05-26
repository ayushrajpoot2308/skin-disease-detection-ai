import matplotlib.pyplot as plt

models = ["VGG16", "ResNet50", "MobileNetV2", "EffNetB0", "DenseNet121", "InceptionV3"]

accuracy = [87.2, 91.4, 88.6, 93.8, 92.5, 91.8]

plt.figure()
plt.bar(models, accuracy)

plt.xlabel("Models")
plt.ylabel("Accuracy (%)")
plt.title("Model Comparison")

plt.xticks(rotation=30)

plt.savefig("static/comparison.png")
plt.show()