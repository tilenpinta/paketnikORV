# PAKETNIK :mailbox:
## RAČUNALNIŠKI VIDI

Algoritem za razpoznavo obraza z jezikom python :snake: uporabljala se je tudi knjižnica OpenCv.

1. Iz zajetih slik enega uporabnika, shranimo LBPH (Local binary pattern histogram) za vsako sliko ter ob tem zapišemo še njegovo uporabniško ime

2. Te histograme in uporabniška imena lahko hranimo v bazi. Trenutno so v (data.json)

3. Ko hočemo razpoznati osebo, ponovno zajememo sliko ter naredimo histogram

4. Potem ta histogram primerjamo s tistim iz baze (compareHistogram s pomočjo metode Chi-square)

5. Preberemo uporabniško ime iz tistega histograma, ki vrne najboljše rezultate.

### RAZVIJALCI :godmode:
  - Ivan Jovanović
  - Tilen Pintarić
  - Daniel Kerec
