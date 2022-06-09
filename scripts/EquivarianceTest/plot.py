import matplotlib.pyplot as plt
import os, glob, json

nets = {
    "LGN" : "./data/LGN/",
    "P-CNN": "./data/P-CNN/",
    "PFN" : "./data/PFN/",
    "ParticleNet" : "./data/ParticleNet/",
    "ResNeXt" : "./data/ResNeXt/",
    "EGNN" : "./data/EGNN/",
    "LorentzNet" : "./data/LorentzNet/"
}

# fig = plt.figure(figsize=(4, 3))
plt.rcParams.update({'font.size': 14})
theme_color = ['xkcd:black', '0.65','xkcd:white'] # text, grid line (grey scale)
plt.grid('on', linestyle='--', color=theme_color[1])
for net, odir in nets.items():
    files = glob.glob(f"{odir}/*.json")
    res = []
    for file in files:
        with open(file, "r") as f:
            res.append(json.loads(f.read()))
    res.sort(key=lambda x:x['beta'])
    beta = [a['beta'] for a in res]
#     loss = [a['loss'] for a in res]
    acc = [a['acc'] for a in res]
    if net == "LorentzNet":
        plt.plot(beta,acc,'.',color='#4bade9',label = net)
    else:
        plt.plot(beta,acc,'.',label = net)
plt.ylim([0.45,1])
plt.xlabel(r"$\beta=\frac{v}{c}$")
plt.ylabel('Accuracy')
# plt.title("Test Accuracy after Lorentz Boosts")
plt.legend(prop={'size':12})

# plt.xlim([0,1])
fig_dir = "./figures/"
os.makedirs(fig_dir, exist_ok=True)
plt.savefig(fig_dir + "EquivarianceTest.jpg", transparent=True, dpi=400, bbox_inches = 'tight')
plt.savefig(fig_dir + "EquivarianceTest.pdf", transparent=True, dpi=400, format='pdf', bbox_inches = 'tight')
plt.show()