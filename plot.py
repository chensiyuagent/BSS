import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15, 'font.serif': "Times New Roman"})

plt.plot([0.26, 0.76, 0.84, 0.96], [0.39, 0.26, 0.21, 0.15],  label='edge prediction')
plt.plot([0.95, 0.75, 0.63, 0.37], [0.92, 0.96, 0.97, 0.98],  label='non-edge prediction')

plt.text(0.26, 0.39,'0.8')
plt.text(0.76, 0.26,'0.6')
plt.text(0.84, 0.21,'0.4')
plt.text(0.96, 0.15,'0.2')
plt.text(0.95, 0.92,'0.8')
plt.text(0.75, 0.96,'0.6')
plt.text(0.63, 0.97,'0.4')
plt.text(0.37, 0.98,'0.2')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()