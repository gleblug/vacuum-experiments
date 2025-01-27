# Работа с BERTAN через COM-port

## Команды

0. `*IDN?` -- сведения о приборе
0. `CONF <MAX_HV>` -- установка максимального напряжения в конфиг
0. `READ?` -- чтение напряжения с BERTAN
0. `SET <HV>` -- установка напряжения на BERTAN
0. `POWER ON/OFF` -- вкл/выкл высокое напряжение

## Примеры

```
> *IDN?
< BERTAN HIGH VOLTAGE arduino analog control v0.1
>
> CONF 10000
> SET 1000
> POWER ON
> READ?
< 1012.3
>
> POWER OFF
> READ?
< 0.0
> READ?
< 5.7
```
