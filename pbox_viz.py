import matplotlib.pyplot as plt

pbox_default = [0,16,32,48,1,17,33,49,2,18,34,50,3,19,35,51,
                    4,20,36,52,5,21,37,53,6,22,38,54,7,23,39,55,
                    8,24,40,56,9,25,41,57,10,26,42,58,11,27,43,59,
                    12,28,44,60,13,29,45,61,14,30,46,62,15,31,47,63]

trivial_pbox = [i for i in range(64)]
print(trivial_pbox)

fig, ax = plt.subplots(figsize=(20, 4))

# Top row: input indices
for i in range(64):
    ax.text(i, 1, str(i), ha='center', va='center', fontsize=8, bbox=dict(facecolor='lightblue', edgecolor='black'))

# Bottom row: ordered numbers 0-63
for i in range(64):
    ax.text(i, 0, str(i), ha='center', va='center', fontsize=8, bbox=dict(facecolor='lightgreen', edgecolor='black'))

# Draw arrows from top index to its pbox mapping in the bottom row
for i in range(64):
    ax.annotate(
        '', xy=(trivial_pbox[i], 0.15), xytext=(i, 0.85),
        arrowprops=dict(arrowstyle='->', color='red', lw=0.8)
    )

ax.set_xlim(-1, 64)
ax.set_ylim(-0.5, 1.5)
ax.axis('off')
plt.title("P-box Bit Mapping: Top = input indices, Bottom = ordered bits")
plt.show()
