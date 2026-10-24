> **Vigencia documental — 18/10/2026:** fuente histórica o de alcance parcial. Sus planes, comandos y afirmaciones no definen por sí solos el sistema actual. Consultar [contraste/índice](../Documentation/auditoria/CONTEXTO_UNSURE.md) y [estado vigente](../Documentation/auditoria/00_ESTADO.md) antes de proponer cambios. El contenido original se conserva debajo.

# Análisis del Reality Check Agresivo

## Problema Identificado

El sistema está aplicando un reality check demasiado agresivo (-25.4 pts) en una predicción para el partido Washington Wizards vs Brooklyn Nets.

### Datos del problema:
- **Predicción base del modelo**: 237 pts
- **Reality check penalty**: -25.4 pts (optimismo extremo)
- **Predicción final**: 241 pts (después de aplicar un clamp de +4 pts)
- **Gap detectado**: +69.0 pts entre predicción y proyección de pace
- **Estado del partido**: Q3 (50% completado) con 116 puntos ya anotados

## Causa Raíz

El problema está en el método `_calcular_proyeccion_pace` en `metamodel/outlier_detector.py` (línea 174):

```python
proyeccion_base = puntos_actuales + (minutos_restantes * pace_actual)
```

### Error conceptual:
- `pace_actual` es 103.84 posesiones por 48 minutos (no puntos por minuto)
- El cálculo está tratando el pace como si fuera puntos por minuto
- Esto da una proyección irrealista: 116 + (24 * 103.84) = 2,612 pts

### Cálculo correcto:
El pace (posesiones por 48 minutos) debe convertirse a puntos por minuto:
1. Pace = 103.84 posesiones/48min ≈ 2.163 posesiones/minuto
2. Eficiencia ofensiva típica en NBA: ~1.1 puntos/posesión
3. Puntos por minuto = 2.163 * 1.1 ≈ 2.38 puntos/minuto

## Solución Propuesta

Modificar el método `_calcular_proyeccion_pace` para convertir correctamente el pace a puntos por minuto:

```python
def _calcular_proyeccion_pace(self, contexto_analizado, q_scores=None, quarter_stage=None):
    # ... código existente ...
    
    # Convertir pace (posesiones/48min) a puntos por minuto
    # Pace típico NBA: 100-105 posesiones por 48 minutos
    # Eficiencia ofensiva típica: 1.05-1.15 puntos por posesión
    eficiencia_ofensiva = 1.10  # Valor promedio de la liga
    puntos_por_minuto = (pace_actual / 48.0) * eficiencia_ofensiva
    
    # Proyección corregida
    proyeccion_base = puntos_actuales + (minutos_restantes * puntos_por_minuto)
    
    # ... resto del código ...
```

## Impacto de la Corrección

Con la corrección:
- Pace de 103.84 → 2.38 puntos/minuto (en lugar de 103.84 pts/minuto)
- Proyección realista: 116 + (24 * 2.38) ≈ 173 pts
- Gap corregido: 237 - 173 = +64 pts (todavía alto pero más realista)
- El reality check seguirá activándose pero con penalizaciones más moderadas

## Implementación

La corrección debe implementarse en:
1. `metamodel/outlier_detector.py` - método `_calcular_proyeccion_pace`
2. Considerar ajustar los umbrales de detección de "optimismo_extremo" (actualmente >20 pts)
3. Evaluar si el factor de ajuste para Q3 (0.92, -8%) es apropiado

## Validación

Después de la corrección:
1. Probar con el mismo caso (WAS vs BKN)
2. Verificar que la proyección de pace sea realista
3. Confirmar que el reality check aplica penalizaciones moderadas
4. Asegurar que las predicciones finales sean más coherentes con el estado del partido