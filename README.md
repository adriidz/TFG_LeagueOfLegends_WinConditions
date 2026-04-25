# TFG

Estructura actual del repositorio:

- `ProgresoActual/`: reinicio support-only con regresion continua, scorer rapido, trainer MLP, referencias de campeones e informe editable.
- `PropuestaInicial/`: archivo del planteamiento anterior, fases de clasificacion, resultados, figuras y modelos historicos.
- `data/` y `data_new/`: datos y caches reutilizables. Se mantienen en raiz para evitar duplicar archivos pesados y romper rutas.
- `src/`: codigo historico que todavia no se ha clasificado por completo. Los scripts minimos del reinicio ya estan copiados/movidos a `ProgresoActual/src/`.

Punto de entrada recomendado:

```powershell
Get-Content ProgresoActual\README.md
```
