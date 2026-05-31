# TFG

Estructura actual del repositorio:

- `ProgresoActual/`: reinicio support-only con regresion continua, scorer rapido, trainer MLP, referencias de campeones e informe editable.
- `ProgresoActual2/`: sandbox reciente para geometria v5 y variantes quantile de la etiqueta support.
- `PropuestaInicial/`: archivo documental del planteamiento anterior. Los artefactos pesados historicos se mantienen fuera de Git.
- `data/` y `data_new/`: datos y caches locales reutilizables, ignorados por Git para evitar duplicar archivos pesados.
- `src/01_data_collection/`: colector raw heredado que sigue siendo util para actualizar datos.

Punto de entrada recomendado:

```powershell
Get-Content ProgresoActual\README.md
```

## Entorno Python

El entorno local recomendado es Python 3.11. El `requirements.txt` fija una pila
cientifica historica (`numpy==1.24.4`, `scipy==1.10.1`, `scikit-learn==1.3.2`,
`matplotlib==3.7.5`) que no es compatible con Python 3.13.

En Windows:

```powershell
py -0p
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r .\requirements.txt
```

Si `py -0p` solo muestra Python 3.13, instala Python 3.11 y recrea `.venv`
antes de instalar dependencias.
