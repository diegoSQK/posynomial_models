from posynomial import *
import matplotlib.pyplot as plt
inf = np.inf

def plots(num_samples):
    a1 = np.linspace(-2, 2, 5)
    a2 = np.linspace(-2, 2, 5)
    a3 = np.linspace(-2, 2, 5)

    a = cartesian([a1,a2,a3])
    w = np.matrix(np.random.rand(3)*10).T
    y = np.array(f(w))
    for i in range(num_samples):
        samp = np.matrix(np.random.rand(3)*10).T
        w = np.c_[w, samp]
        out = f(samp)
        y = np.append(y, out)

    lambdas = [n*np.ones(len(a)) for n in np.logspace(0, 5, 20)]
    errors = []
    cards = []
    for l in lambdas:
        fhat = recover_posynomial(w, y, a, l)
        coeffs = [fhat[i] for i in range(len(fhat)) if fhat[i] > 0]
        exps = [a[i] for i in range(len(fhat)) if fhat[i] > 0]

        diffs = []
        max_y = float(norm(y, inf))
        for j in range(w.shape[0]):
            y_est = posynomial(coeffs, exps, w[:,j])
            diffs.append(abs(y_est - y[j])/max_y)

        errors.append(max(diffs))
        cards.append(norm(fhat, 0))

    plt.plot(np.logspace(0, 5, 20), errors)
    plt.xscale('log')
    plt.xlabel("Lambda")
    plt.ylabel("Error")
    plt.savefig("lambda_vs_err.jpg")
    plt.clf()
    plt.plot(np.logspace(0, 5, 20), cards)
    plt.xscale('log')
    plt.xlabel("Lambda")
    plt.ylabel("Cardinality")
    plt.savefig("lambda_vs_card.jpg")

