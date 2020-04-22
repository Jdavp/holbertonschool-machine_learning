#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

persons = ['Farrah', 'Fred', 'Felicia']
fruits_list = ['apples', 'bananas', 'oranges', 'peaches']
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

for x in range(len(fruit)):
    plt.bar(persons, fruit[x], color=colors[x], label=fruits_list[x],
            width=0.5, bottom=np.sum(fruit[:x], axis=0))

plt.legend()
plt.ylabel('Quantity of Fruit')
plt.title('Number of Fruit per Person')
plt.yticks(np.arange(0, 81, 10))
plt.show()
