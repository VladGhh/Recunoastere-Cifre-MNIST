# Recunoastere Cifre MNIST
Am construit această rețea de la zero pentru că am vrut să înțeleg cum funcționează un AI. Fără TensorFlow, fără PyTorch — doar Python, NumPy și multă matematică aplicată. Programul învață să recunoască cifre scrise de mână (setul MNIST) și ajunge la o precizie de aproximativ 94-95% în funcție de cât timp are să "învețe". Am aflat astfel cât de utile au fost cursurile de algebră liniară și analiză din facultate.

## Cum funcționează matematica?

Când primește o imagine, algoritmul o codifică într-un vector, fiecare pixel primind o valoare reprezentând cât de deschis (apropiat de a fi alb) este. Tot programul se bazează pe minimizarea erorii prin Stochastic Gradient Descent (SGD). Practic, rețeaua caută "cea mai adâncă vale" sau punctul de minim al funcției cost, funcție ce măsoară cât de greșit este răspunsul oferit.

### 1. Activarea (Sigmoid)
Fiecare neuron folosește funcția Sigmoid pentru a decide cât de mult "se aprinde". Aceasta transformă orice număr într-o valoare între 0 și 1:
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

### 2. Backpropagation
Ca să știm cum să modificăm rețeaua, trebuie să aflăm cât a contribuit fiecare neuron la eroarea finală. 

În centrul calculelor stă **$z$**, care reprezintă **intrarea ponderată** a unui neuron (adică suma semnalelor primite de la stratul anterior, înainte de a trece prin funcția Sigmoid):
$$z = w \cdot a + b$$

Esența algoritmului este calcularea variabilei **$\delta$ (Delta)**. Matematic, $\delta$ nu este doar o eroare oarecare, ci este **derivata funcției de cost în raport cu $z$** ($\frac{\partial C}{\partial z}$). Ea ne spune exact cât de repede se schimbă eroarea totală dacă modificăm foarte puțin intrarea $z$ a unui neuron.

Pentru ultimul strat, formula pe care am implementat-o este:
$$\delta^L = (a^L - y) \* \sigma'(z^L)$$

Odată ce avem acest $\delta$, restul e algebră simplă pentru a afla gradienții:
- **Gradient Bias:** $\frac{\partial C}{\partial b} = \delta$
- **Gradient Weights:** $\frac{\partial C}{\partial w} = \delta \cdot a_{anterior}^T$

Practic, $\delta$ funcționează ca o busolă: ne arată direcția și intensitatea cu care trebuie să "împingem" fiecare weight și bias pentru ca, la următoarea imagine, eroarea să fie mai mică.

Odată ce am aflat acest $\delta$, am putut calcula exact gradienții pentru a actualiza parametrii:
- **Pentru Bias:** Gradientul este pur și simplu $\delta$.
- **Pentru Weights:** Gradientul este $\delta \cdot a_{anterior}^T$ (eroarea curentă înmulțită cu semnalul primit din spate).

Practic, fiecare greutate se modifică în funcție de cât de mult a contribuit la eroarea finală. Este un sistem de auto-corecție matematică care funcționează ca o busolă pentru rețea.


## Surse de inspirație
- Structura matematică și intuiția vizuală: Grant Sanderson (3Blue1Brown) - Neural Networks series.

## Cum poate fi testat?
1. Desenează o cifră în Paint (28x28 pixeli, fundal negru, scris alb) și salveaz-o ca `cifra.png`. În proiect am adăugat deja un astfel de exemplu pentru testare.
2. Instalează ce ai nevoie: `pip install numpy Pillow`.
3. Rulează codul: `python neural_network.py`.
