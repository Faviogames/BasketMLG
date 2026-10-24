> **Vigencia documental — 18/10/2026:** fuente histórica o de alcance parcial. Sus planes, comandos y afirmaciones no definen por sí solos el sistema actual. Consultar [contraste/índice](../Documentation/auditoria/CONTEXTO_UNSURE.md) y [estado vigente](../Documentation/auditoria/00_ESTADO.md) antes de proponer cambios. El contenido original se conserva debajo.

# 🧠 Guía del Sistema de Aprendizaje del Metamodelo

## 📋 Tabla de Contenidos
1. [Resumen de la Implementación](#resumen-de-la-implementación)
2. [¿Qué se ha Activado?](#qué-se-ha-activado)
3. [¿Dónde se Guardan las Predicciones?](#dónde-se-guardan-las-predicciones)
4. [Cómo Registrar Resultados Manualmente](#cómo-registrar-resultados-manualmente)
5. [Ejemplo Práctico](#ejemplo-práctico)
6. [Cómo Funciona el Aprendizaje](#cómo-funciona-el-aprendizaje)
7. [Scripts de Ayuda](#scripts-de-ayuda)
8. [Beneficios a Largo Plazo](#beneficios-a-largo-plazo)

---

## 🎯 Resumen de la Implementación

El sistema de aprendizaje del metamodelo ha sido **completamente activado** y ahora puede:

1. **Registrar resultados finales** de partidos para aprendizaje
2. **Analizar la efectividad** de los ajustes contextuales
3. **Optimizar parámetros** basándose en resultados históricos
4. **Mejorar predicciones futuras** mediante aprendizaje continuo

---

## ✅ ¿Qué se ha Activado?

### 1. Sistema de Aprendizaje en MetaModeloInteligente

En [`metamodel/meta_model.py`](metamodel/meta_model.py:247-252) se ha activado el aprendizaje:

```python
# ANTES (comentado):
# if self.config['learning']['enabled']:
#     self.sistema_aprendizaje.registrar_resultado(...)

# AHORA (activado):
if self.config['learning']['enabled']:
    # Guardar información para aprendizaje posterior
    info_ajuste['aprendizaje_pendiente'] = True
    info_ajuste['timestamp_aprendizaje'] = datetime.now().isoformat()
```

### 2. Nuevos Métodos para Registro Manual

Se han agregado 4 nuevos métodos a la clase `MetaModeloInteligente`:

#### a) `registrar_resultado_manual()`
Registra un resultado usando el ID único de la predicción:
```python
metamodelo.registrar_resultado_manual('ID_PREDICCIÓN', resultado_real)
```

#### b) `cargar_y_registrar_resultado()`
Carga una predicción desde archivo y registra el resultado:
```python
metamodelo.cargar_y_registrar_resultado('ruta/archivo.json', resultado_real)
```

#### c) `mostrar_info_predicciones_guardadas()`
Muestra todas las predicciones guardadas para facilitar el registro:
```python
metamodelo.mostrar_info_predicciones_guardadas()
```

#### d) `_actualizar_parametros_aprendizaje()`
Actualiza automáticamente los parámetros del metamodelo cuando hay suficientes datos.

---

## 📁 ¿Dónde se Guardan las Predicciones?

Las predicciones con contexto completo se guardan en:

```
prediction_records/daily/
├── ID_UNICO_Equipos_vs_Equipos_Fecha.json    # Predicciones individuales
└── YYYY-MM-DD_predictions.json              # Resumen diario
```

### Estructura del Archivo de Predicción Individual

Cada archivo contiene:
```json
{
  "id_unico": "5F9B72B6",
  "home_team": "Cleveland Cavaliers",
  "away_team": "Memphis Grizzlies",
  "fecha": "2025-11-16",
  "prediccion_base": 206.0,
  "prediccion_final": 203.4,
  "contexto_analizado": { ... },
  "outlier_info": { ... },
  "timestamp": "2025-11-16T01:10:09.324Z"
}
```

---

## 📝 Cómo Registrar Resultados Manualmente

### Opción 1: Usando el Script Interactivo

Ejecuta el script `registrar_resultado.py`:
```bash
python registrar_resultado.py
```

Este script te mostrará:
1. Lista de predicciones recientes
2. Selección interactiva
3. Ingreso del resultado final
4. Confirmación antes de registrar

### Opción 2: Usando Código Directamente

```python
from metamodel.meta_model import MetaModeloInteligente

# Inicializar metamodelo
metamodelo = MetaModeloInteligente()

# Opción A: Usando ID único
metamodelo.registrar_resultado_manual('5F9B72B6', 215)

# Opción B: Usando archivo completo
metamodelo.cargar_y_registrar_resultado(
    'prediction_records/daily/5F9B72B6_Cavaliers_vs_Grizzlies_2025-11-16.json',
    215
)
```

### Opción 3: Ver Predicciones Disponibles

```python
# Mostrar todas las predicciones guardadas
metamodelo.mostrar_info_predicciones_guardadas()
```

---

## 🏀 Ejemplo Práctico

Usando el caso real del log proporcionado:

```python
# 1. Inicializar metamodelo
metamodelo = MetaModeloInteligente(league_name="NBA")

# 2. Registrar resultado del partido Cleveland vs Memphis
resultado_real = 215  # Ejemplo: el partido terminó con 215 puntos totales
id_prediccion = "5F9B72B6"  # ID del ejemplo

# 3. Registrar para aprendizaje
resultado = metamodelo.registrar_resultado_manual(id_prediccion, resultado_real)

# 4. Ver resultado del registro
print(f"✅ Registro exitoso: {resultado['success']}")
print(f"📈 Error Base: {resultado['error_base']:.1f} pts")
print(f"📈 Error Ajustada: {resultado['error_ajustada']:.1f} pts")
print(f"🎯 Ajuste Beneficioso: {'Sí' if resultado['ajuste_beneficioso'] else 'No'}")
```

---

## 🧠 Cómo Funciona el Aprendizaje

### 1. Análisis de Efectividad de Ajustes

El sistema compara:
- **Predicción Base**: Sin ajustes contextuales
- **Predicción Ajustada**: Con ajustes contextuales aplicados
- **Resultado Real**: Puntos finales del partido

Determina si los ajustes fueron beneficiosos:
```python
error_base = abs(prediccion_base - resultado_real)
error_ajustada = abs(prediccion_ajustada - resultado_real)
ajuste_beneficioso = error_ajustada < error_base
```

### 2. Registro de Patrones

Guarda información contextual:
- Cuarto del partido
- Minutos restantes
- Situación de garbage time
- Intensidad del partido
- Tipo de outlier detectado

### 3. Optimización de Parámetros

Cuando hay ≥10 registros en el historial:
- Analiza patrones de éxito/fracaso
- Optimiza pesos de ajustes contextuales
- Ajusta umbrales de detección
- Mejora factores de aprendizaje

### 4. Mejora Continua

Con cada resultado registrado:
- El modelo aprende qué ajustes funcionan mejor
- Reconoce sesgos sistemáticos
- Se adapta a cambios en la liga
- Mejora predicciones futuras

---

## 🛠️ Scripts de Ayuda

### 1. `demo_aprendizaje_metamodelo.py`

Demostración completa del sistema de aprendizaje:
```bash
python demo_aprendizaje_metamodelo.py
```

Muestra:
- Cómo inicializar el metamodelo con aprendizaje
- Dónde se guardan las predicciones
- Ejemplos de registro manual
- Explicación detallada del proceso de aprendizaje

### 2. `registrar_resultado.py`

Interfaz interactiva para registrar resultados:
```bash
python registrar_resultado.py
```

Características:
- Lista predicciones recientes
- Selección numérica intuitiva
- Validación de entrada
- Confirmación antes de registrar
- Muestra errores y mejoras

---

## 📈 Beneficios a Largo Plazo

### 1. **Predicciones Más Precisas**
- El modelo aprende de errores pasados
- Ajustes contextuales más efectivos
- Mejor calibración para situaciones específicas

### 2. **Detección Mejorada de Outliers**
- Reconoce outliers legítimos vs errores del modelo
- Ajusta umbrales según la liga y contexto
- Reduce falsos positivos/negativos

### 3. **Adaptación a Cambios**
- Se adapta a cambios en el estilo de juego
- Reconoce nuevas tendencias en la liga
- Mantiene relevancia a lo largo del tiempo

### 4. **Sistema Autónomo**
- Requiere mínima intervención manual
- Optimización automática de parámetros
- Aprendizaje continuo sin supervisión

---

## 🚀 Próximos Pasos

1. **Ejecuta predicciones en vivo** para generar datos
2. **Registra los resultados finales** cuando terminen los partidos
3. **Monitorea el progreso** después de 10+ registros
4. **Observa la mejora** en predicciones futuras

---

## 📞 Soporte

Si tienes preguntas o problemas:

1. Revisa el log del sistema para ver errores
2. Verifica que los archivos de predicciones existan
3. Asegúrate de que el resultado ingresado sea válido
4. Usa `metamodelo.obtener_estadisticas()` para ver el progreso

---

## 🎯 Conclusión

El sistema de aprendizaje del metamodelo está **completamente operativo** y listo para mejorar las predicciones mediante el aprendizaje continuo. Cada resultado registrado hace el sistema más inteligente y preciso.

¡Ahora puedes ayudar al metamodelo a aprender y mejorar! 🏀🧠