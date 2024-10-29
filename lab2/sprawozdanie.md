     
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
- Siła mutacji = 0,55 dla F2
- ilość iteracji = 10000 // Wielkość iteracji

b) Tabele wyników
### Funkcja F2
| Wielkość populacji | Średnia | Odchylenie Standardowe | Min | Max | Ilość iteracji |
|:---------:|:---------:|:------------------------:|:-----:|:------:|:----------:|
|2|463043,75|632555,28|1007,35| 2923758.12| 5000|
|4|11379,37|10945,09|228,89|40709,12|2500|
|8|8592,22|9555,57|226,13|46338,95|1250|
|16|368044,48|1046892,04|311,40|6460105,63|625|
|32|275497684,59|1034552056,17|352,99|5832896945,78|312|
|64|13166629595.36|28371953394.19|1872.44|130173872823.69|156|

### Funkcja F13
| Wielkość populacji | Średnia | Odchylenie Standardowe | Min | Max | Ilość iteracji |
|:---------:|:---------:|:------------------------:|:-----:|:------:|:----------:|
