import matplotlib.pyplot as plt

f = open("training_2021-05-16_19-51-45.csv","r")
data = f.read().strip().split()
f.close()
train_loss, valid_loss, valid_acc = [], [], []
for line in data:
    item = line.strip().split(',')
    a, b, c = eval(item[0]), eval(item[1]), eval(item[2])
    train_loss.append(a)
    valid_loss.append(b)
    valid_acc.append(c)

plt.plot(train_loss)
plt.title('training loss curve')
plt.xlabel('epoch')
plt.ylabel('loss')
# plt.plot(valid_loss)
# plt.plot(valid_acc)
plt.savefig('curve.jpg')

# plt.plot(train_loss)
# plt.title('validation accuracy curve')
# plt.xlabel('epoch')
# plt.ylabel('accuracy')
# plt.plot(valid_loss)
# plt.plot(valid_acc)
# plt.savefig('curve.jpg')

