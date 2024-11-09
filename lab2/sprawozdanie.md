     
# Sprawozdanie WSI Lab2 Tobiasz Kownacki
--------------------
## Zadanie:
- Zaimplementować klasyczny algorytm ewolucyjny z selekcją turniejową i sukcesją generacyjną, bez krzyżowania.
    - Dostępny budżet to 10000 ewaluacji funkcji celu.
    - Optymalizujemy funkcje numer 2 i 13 z CEC 2017 w 10 wymiarach.
    - Ograniczenia kostkowe przestrzeni to -100, 100.
- Zbadać wpływ:
    - liczby osobników w populacji.
    - siły mutacji.
----------------------
 **Wszystkie dane przedstawiają wyniki z 50 uruchomień algorytmu z zastosowanymi parametrami**

## 1. Liczba osobników w populacji
a) Parametry algorytmu:
- Siła mutacji = 0,7 dla F2 i 0,1 dla F13
- ilość iteracji = 10000 // Wielkość populacji
- liczba wymiarów = 10

b) Tabele wyników
### Funkcja F2
| Wielkość populacji | Średnia | Odchylenie Standardowe | Min | Max | Ilość iteracji |
|:---------:|:---------:|:------------------------:|:-----:|:------:|:----------:|
|2| 37157,81| 87688,23| 288,50| 552392,51| 5000|
|4| 6092,92| 7911,06| 230,97| 26585,33| 2500|
|8| **4042,85**| 4461,96| **228,09**| 18422,26| 1250|
|16| 19841,37| 32236,74| 249,91| 185413,12| 625|
|32| 1301873062,04| 6570854693,30| 312,84| 45176009681,98| 312|

Najniższe minimum oraz średnią znaleźliśmy dla populacji składającej się z 8 osobników. Znalezione minimum wynoszące 228,09 jest bardzo bliskie tego prawdziwego wynoszącego 200.
Średnia oraz odchylenie standardowe jest dość wysokie, przez co dla uzyskania w miarę dobrego wyniku należy uruchomić algorytm kilkanaście razy.
   

   
### Funkcja F13
| Wielkość populacji | Średnia | Odchylenie Standardowe | Min | Max | Ilość iteracji |
|:---------:|:---------:|:------------------------:|:-----:|:------:|:----------:|
|2|18013,08|13338,21| 2054,92| 62356,19| 5000|
|4|17116,78| 14096,51| 1960,03| 61853,68| 2500|
|8|**13663,97**| 11634,72| 2395,94| 64386,43| 1250|
|16|116762939,74| 428699080,30| **1748,77**| 2290596016,41|625|
|32|65694359,28| 220812681,09| 2162,20| 1079489505,78| 312|

Najniższe minimum znaleziono dla populacji składającej się z 16 osobników. Najniższą średnią znaleziono jednak dla populacji z 8 osobnikami.   
Dla populacji z najniższym minimum, średnia, maximum oraz odchylenie standardowe jest o wiele większe niż dla populacji z 8 osobnikami.
Dla uzyskania dobrego wyniku również należy uruchomić algorytm kilkanaście razy


## 2. Siły mutacji
a) Parametry algorytmu:
- populacja = 8 dla F2 i 16 dla F13
- ilość iteracji = 1250 dla F2 i 625 dla F13
- liczba wymiarów = 10

b) Tabele wyników
### Funkcja F2
| Siła mutacji | Średnia | Odchylenie Standardowe | Min | Max |
|:---------:|:---------:|:------------------------:|:-----:|:------:|
|0,3|1220429,11| 4220197,76| 276,89| 23137042,88|
|0,7| **5158,38**| 5999,02| **225,08**| 23289,04|
|1| 7866,51| 8034,99| 318,17| 27057,42|
|2| 113926,65| 212981,83| 2328,81| 1105899,87|
|3|373952,13| 507776,50| 8152,65| 2888955,42|

Najmniejsze minimum oraz średnią znaleziono dla siły mutacji równej 0.7 Mniejsze wartości powodują, że punktom trudno jest dostać się do minimum, bo poruszają się za wolno.
Większe siły mutacji od 0,7 powodują, że punkty mają problem z znalezieniem dokładnego minimum.

### Funkcja F13
| Siła mutacji | Średnia | Odchylenie Standardowe | Min | Max |
|:---------:|:---------:|:------------------------:|:-----:|:------:|
|0,01| 4314106825,92| 4088310385,44| 14464,57| 16597495238,68|
|0,05| 768114436,45| 2274688913,63| 2150,97| 12369686745,34|
|0,1| 22057395,54| 148041890,30| **1689,79**| 1057437081,29|
|0,2| 15374,86| 12121,60| 2063,65| 47388,09|
|0,5| **13447,72**| 10111,66| 2306,40| 40561,73|
|1| 23555.16| 14894.60| 3756.78| 68429.11|
|2|31140.42| 15955.26| 3282.54| 65265.81|

Najmniejsze minimum znaleziono dla siły mutacji równej 0,1, ale najniższą średnią oraz odchylenie standardowe dla 0,5. Sigma równa 0,5 daje średnio o wiele lepsze wyniki niż dla siły mutacji wynoszącej 0,1.

### 3. Zwiększenie budżetu do 50 000

- **F2**, populacja = 8, siła mutacji = 0,7

| Budżet | Średnia | Odchylenie Standardowe | Min | Max | Ilość iteracji |
|:---------:|:---------:|:------------------------:|:-----:|:------:|:----------:|
|10000|6061,60| 6764,05| 241,56| 21240,24| 1250|
|50000| 283,32| 58,93| 207,52| 429,46| 6250|

- **F13**, populacja = 16, siła mutacji = 0,1

| Budżet | Średnia | Odchylenie Standardowe | Min | Max | Ilość iteracji |
|:---------:|:---------:|:------------------------:|:-----:|:------:|:----------:|
|10000|22411770.40| 156764313.58| 1717.51| 1119761962.51| 625|
|50000|17235,77| 12394,93| 1650,69| 49329,14| 3125|

Zwiększenie budżetu do 50000 w obu funkcjach znacznie pomaga algorytmowi w znalezieniu lepszego minimum.
 W przypadku F2, minimum znacznie się zbliżyło do prawdziwego minimum równego 200. Średnia i odchylenie standardowe spadło diametralnie. Podwyższenie budżetu znacząco poprawia niezawodność algorytmu przy pojedynczych uruchomieniach optymalizacji funkcji F2.
 W przypadku F13, minimum spadło nieznacznie, ale średnia oraz odchylenie standardowe spadło diametralnie. Zwiększenie budżetu spowodowało poprawę niezawodności algorytmu przy pojedynczym uruchomieniu, ale ta niezawodność jest o wiele niższa niż w przypadku F2

 ### 4. Wnioski
 Algorytm ewolucyjny potrafi znaleźć bardzo bliskie okolice minimum globalnego, przy odpowiedniej sile mutacji oraz liczbie osobników. Czasami jest potrzeba zwiększenia budżetu w przypadku bardziej skomplikowanych funkcji, aby uzyskać lepsze wyniki. Szczególna poprawa jest w przypadku średniej wyników. Algorytm Ewolucyjny jest dość prosty w implementacji. W porównaniu do metody gradientu prostego, algorytm potrafi znaleźć optima globalne. Średnie wyników również są na korzyść dla algorytmu ewolucyjnego. Uzyskanie wiarygodnego wyniku algorytmu ewolucyjnego wymaga wielokrotnego uruchomienia algorytmu przy odpowiednio dużym budżecie.