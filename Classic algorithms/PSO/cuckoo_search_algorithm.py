import math
import random


class Cuckoo:

    def __init__(self,x,y):
        self.x = x
        self.y = y

    def easom_function(self):
        return -math.cos(self.x)*math.cos(self.y)*math.exp(-((self.x-math.pi)**2+(self.y-math.pi)**2))


iteration = 1000
cuckoos = []

# Tworzymy początkową populację kukułek z losowymi współrzędnymi
for i in range(1000):
    new_cuckoo = Cuckoo(random.uniform(3, 10), random.uniform(3, 10))
    cuckoos.append(new_cuckoo)

# Główna pętla
for j in range(iteration):
    alpha = 0.2
    # Przesunięcie kukułek
    for k in range(1000):
        cuckoos[k].x = cuckoos[k].x + (math.sqrt(1 / (math.pi)) * math.exp(-1 / ((cuckoos[k].x - 0.3)))/(math.pow(cuckoos[k].x - 0.3, 1.5))) * alpha
        cuckoos[k].y = cuckoos[k].y + (math.sqrt(1 / (math.pi)) * math.exp(-1 / ((cuckoos[k].y - 0.3)))/(math.pow(cuckoos[k].y - 0.3, 1.5))) * alpha
        # Decyzja gospodarza
        # Tworzymy nową kukułke do porównania
        new_cuckoo = Cuckoo(random.uniform(3, 10), random.uniform(3, 10))
        if (random.uniform(0, 1) < 0.5 and new_cuckoo.easom_function() < cuckoos[k].easom_function()):
            # Jeśli losowa liczba z przedziału (0,1) jest mniejsza od 0.5 to usuwamy kukułkę i na jej miejsce
            # pojawia się nowa. Tak samo gdy nowa kukułka jest lepsza od starej.
            cuckoos[k] = new_cuckoo

# Szukamy najlepszej kukułki
for p in range(1000):
    value = cuckoos[p].easom_function()
    if p == 0:
        current_min_value = value
        current_min_cuckoo = cuckoos[p]
    else:
        if current_min_value > value:
            current_min_value = value
            current_min_cuckoo = cuckoos[p]

print('x = '+str(round(current_min_cuckoo.x,3)))
print('y = '+str(round(current_min_cuckoo.y,3)))
print('f(x,y) = '+str(round(current_min_value,3)))
