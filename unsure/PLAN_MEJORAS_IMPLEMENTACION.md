> **Vigencia documental — 18/10/2026:** fuente histórica o de alcance parcial. Sus planes, comandos y afirmaciones no definen por sí solos el sistema actual. Consultar [contraste/índice](../Documentation/auditoria/CONTEXTO_UNSURE.md) y [estado vigente](../Documentation/auditoria/00_ESTADO.md) antes de proponer cambios. El contenido original se conserva debajo.

# Plan de Implementación de Mejoras - Sistema de Predicciones

## 🎯 Resumen Ejecutivo

Plan detallado para implementar las mejoras identificadas en el análisis de predicciones, enfocado en corregir los patrones de fallo del modelo base y mejorar la efectividad del meta-modelo de reality check.

## 📋 Prioridades de Implementación

### 🚨 Críticas (Implementar esta semana)
1. **Detección mejorada de outliers legítimos**
2. **Perfiles de equipos alto-scoring**
3. **Ajuste de umbrales del Reality Check**

### ⚡ Alta (Próximas 2 semanas)
1. **Sistema de contexto por equipos**
2. **Ajustes dinámicos por etapa del partido**
3. **Validación histórica mejorada**

### 🔄 Media (Próximo mes)
1. **Sistema ensemble de Reality Checks**
2. **Aprendizaje continuo automático**
3. **Dashboard de monitoreo**

## 🔧 Implementación Crítica - Semana 1

### 1. Mejor Detector de Outliers Legítimos

#### Código Propuesto
```python
# metamodel/improved_outlier_detector.py

class ImprovedOutlierDetector:
    """
    Detector mejorado que distingue entre outliers legítimos y optimismo sistemático
    """
    
    def __init__(self, league_name: str, config: Dict[str, Any]):
        self.league_name = league_name
        self.config = config
        self.team_profiles = self._load_team_profiles()
        
    def _load_team_profiles(self) -> Dict[str, Dict]:
        """
        Carga perfiles específicos de equipos
        """
        return {
            'Miami Heat': {
                'style': 'high_scoring',
                'avg_ppg': 115.2,
                'avg_opponent_ppg': 112.8,
                'outlier_threshold': 280,
                'rc_intervention_threshold': 285
            },
            'New York Knicks': {
                'style': 'high_scoring',
                'avg_ppg': 118.5,
                'avg_opponent_ppg': 114.3,
                'outlier_threshold': 285,
                'rc_intervention_threshold': 290
            },
            'Phoenix Suns': {
                'style': 'high_scoring',
                'avg_ppg': 116.8,
                'avg_opponent_ppg': 113.2,
                'outlier_threshold': 275,
                'rc_intervention_threshold': 280
            },
            'Utah Jazz': {
                'style': 'offensive_variable',
                'avg_ppg': 112.3,
                'avg_opponent_ppg': 116.7,
                'outlier_threshold': 265,
                'rc_intervention_threshold': 270
            }
        }
    
    def detectar_outlier_mejorado(
        self,
        prediccion_base: float,
        equipos: List[str],
        contexto: Dict[str, Any],
        q_scores: Dict[str, int] = None
    ) -> Dict[str, Any]:
        """
        Detección mejorada de outliers con contexto específico
        """
        resultado = {
            'es_outlier': False,
            'tipo': None,
            'confianza': 0.0,
            'factores': [],
            'recomendacion': None
        }
        
        # 1. Verificar perfiles de equipos alto-scoring
        equipos_alto_scoring = [
            equipo for equipo in equipos 
            if equipo in self.team_profiles and 
            self.team_profiles[equipo]['style'] == 'high_scoring'
        ]
        
        if equipos_alto_scoring:
            max_threshold = max(
                self.team_profiles[equipo]['outlier_threshold']
                for equipo in equipos_alto_scoring
            )
            
            if prediccion_base > max_threshold:
                # Verificar si es outlier legítimo
                if self._es_outlier_legitimo_alto_scoring(
                    prediccion_base, equipos_alto_scoring, contexto
                ):
                    resultado.update({
                        'es_outlier': True,
                        'tipo': 'outlier_legitimo_alto_scoring',
                        'confianza': 0.8,
                        'factores': [f'Equipos alto-scoring: {", ".join(equipos_alto_scoring)}'],
                        'recomendacion': 'mantener_sin_ajuste'
                    })
                    return resultado
        
        # 2. Detección de spikes en Q2
        if q_scores:
            q2_spike = self._detectar_q2_spike(q_scores)
            if q2_spike['es_spike']:
                resultado['factores'].append(q2_spike['descripcion'])
                resultado['confianza'] += 0.3
                
                if prediccion_base > 250:
                    resultado.update({
                        'es_outlier': True,
                        'tipo': 'outlier_legitimo_spike_q2',
                        'confianza': min(0.7, resultado['confianza']),
                        'recomendacion': 'ajuste_conservador'
                    })
        
        # 3. Detección de optimismo sistemático
        if prediccion_base > 240:
            if not self._hay_contexto_justificado(prediccion_base, equipos, contexto):
                resultado.update({
                    'es_outlier': True,
                    'tipo': 'optimismo_sistemático',
                    'confianza': 0.75,
                    'factores': ['Predicción alta sin contexto justificado'],
                    'recomendacion': 'reduccion_significativa'
                })
        
        return resultado
    
    def _es_outlier_legitimo_alto_scoring(
        self,
        prediccion_base: float,
        equipos: List[str],
        contexto: Dict[str, Any]
    ) -> bool:
        """
        Verifica si una predicción alta es un outlier legítimo para equipos alto-scoring
        """
        # 1. Verificar histórico H2H si está disponible
        # (Implementar consulta a base de datos históricos)
        
        # 2. Verificar factores contextuales
        factores_justificativos = 0
        
        # Factor 1: Ambos equipos alto-scoring
        if len(equipos) >= 2:
            factores_justificativos += 2
        
        # Factor 2: Partido en etapa avanzada con ritmo alto
        if contexto.get('pace_actual', 0) > 105:
            factores_justificativos += 1
        
        # Factor 3: Sin señales de desaceleración
        if not contexto.get('intensity_drop', False):
            factores_justificativos += 1
        
        # Requerir al menos 3 factores justificativos
        return factores_justificativos >= 3
    
    def _detectar_q2_spike(self, q_scores: Dict[str, int]) -> Dict[str, Any]:
        """
        Detecta spikes anómalos en el segundo cuarto
        """
        q1_total = q_scores.get('q1_home', 0) + q_scores.get('q1_away', 0)
        q2_total = q_scores.get('q2_home', 0) + q_scores.get('q2_away', 0)
        
        if q1_total <= 0 or q2_total <= 0:
            return {'es_spike': False}
        
        spike = q2_total - q1_total
        
        if spike > 20:  # Spike significativo
            return {
                'es_spike': True,
                'descripcion': f'Spike Q2: +{spike} pts sobre Q1',
                'magnitud': spike,
                'impacto': 'aumento_scoring_proyectado'
            }
        
        return {'es_spike': False}
    
    def _hay_contexto_justificado(
        self,
        prediccion_base: float,
        equipos: List[str],
        contexto: Dict[str, Any]
    ) -> bool:
        """
        Verifica si hay contexto que justifique una predicción alta
        """
        # 1. Equipos con perfil alto-scoring
        equipos_alto_scoring = any(
            equipo in self.team_profiles and 
            self.team_profiles[equipo]['style'] == 'high_scoring'
            for equipo in equipos
        )
        
        if equipos_alto_scoring:
            return True
        
        # 2. Pace excepcionalmente alto
        if contexto.get('pace_actual', 0) > 110:
            return True
        
        # 3. Spike en Q2 detectado
        if contexto.get('q2_spike', False):
            return True
        
        return False
```

### 2. Reality Check Adaptativo

#### Código Propuesto
```python
# metamodel/adaptive_reality_check.py

class AdaptiveRealityCheck:
    """
    Reality Check con umbrales adaptativos basados en contexto
    """
    
    def __init__(self, league_name: str, config: Dict[str, Any]):
        self.league_name = league_name
        self.config = config
        self.base_thresholds = {
            'optimism_threshold': 15.0,
            'intervention_threshold': 5.0,
            'max_adjustment': 20.0
        }
        self.current_thresholds = self.base_thresholds.copy()
    
    def aplicar_ajuste_adaptativo(
        self,
        prediccion_base: float,
        outlier_info: Dict[str, Any],
        equipos: List[str],
        contexto: Dict[str, Any],
        quarter_stage: str = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Aplica ajuste adaptativo basado en contexto específico
        """
        # 1. Ajustar umbrales según contexto
        self._ajustar_umbrales_contexto(equipos, quarter_stage, contexto)
        
        # 2. Determinar si aplicar ajuste
        ajuste = 0.0
        info_ajuste = {
            'aplicado': False,
            'motivo': None,
            'magnitud': 0.0,
            'umbrales_usados': self.current_thresholds.copy()
        }
        
        if outlier_info['es_outlier']:
            tipo_outlier = outlier_info['tipo']
            
            if tipo_outlier == 'optimismo_sistemático':
                # Aplicar reducción significativa
                gap = prediccion_base - contexto.get('proyeccion_realista', prediccion_base)
                ajuste = min(gap * 0.7, self.current_thresholds['max_adjustment'])
                info_ajuste.update({
                    'aplicado': True,
                    'motivo': 'optimismo_sistemático_detectado',
                    'magnitud': ajuste
                })
            
            elif tipo_outlier == 'outlier_legitimo_alto_scoring':
                # NO aplicar ajuste para outliers legítimos
                ajuste = 0.0
                info_ajuste.update({
                    'aplicado': False,
                    'motivo': 'outlier_legitimo_detectado_mantener',
                    'magnitud': 0.0
                })
            
            elif tipo_outlier == 'outlier_legitimo_spike_q2':
                # Ajuste conservador para spikes
                ajuste = self.current_thresholds['intervention_threshold']
                info_ajuste.update({
                    'aplicado': True,
                    'motivo': 'ajuste_conservador_spike_q2',
                    'magnitud': ajuste
                })
        
        prediccion_ajustada = prediccion_base - ajuste
        return prediccion_ajustada, info_ajuste
    
    def _ajustar_umbrales_contexto(
        self,
        equipos: List[str],
        quarter_stage: str,
        contexto: Dict[str, Any]
    ):
        """
        Ajusta dinámicamente los umbrales según el contexto
        """
        # Resetear a valores base
        self.current_thresholds = self.base_thresholds.copy()
        
        # 1. Ajuste por equipos alto-scoring
        equipos_alto_scoring = [
            equipo for equipo in equipos
            if self._es_equipo_alto_scoring(equipo)
        ]
        
        if equipos_alto_scoring:
            self.current_thresholds['optimism_threshold'] += 10
            self.current_thresholds['intervention_threshold'] += 5
            self.current_thresholds['max_adjustment'] += 10
        
        # 2. Ajuste por etapa del partido
        if quarter_stage == 'HT':
            # Más conservador en half-time
            self.current_thresholds['intervention_threshold'] += 3
            self.current_thresholds['optimism_threshold'] += 5
        
        elif quarter_stage in ['q1_progress', 'q2_progress']:
            # Ligeramente más permisivo en etapas tempranas
            self.current_thresholds['optimism_threshold'] += 3
        
        # 3. Ajuste por contexto de ritmo
        if contexto.get('pace_actual', 0) > 105:
            self.current_thresholds['optimism_threshold'] += 8
            self.current_thresholds['max_adjustment'] += 5
        
        # 4. Ajuste por spikes detectados
        if contexto.get('q2_spike', False):
            self.current_thresholds['optimism_threshold'] += 12
            self.current_thresholds['intervention_threshold'] += 8
    
    def _es_equipo_alto_scoring(self, equipo: str) -> bool:
        """
        Verifica si un equipo tiene perfil alto-scoring
        """
        equipos_alto_scoring = [
            'Miami Heat', 'New York Knicks', 'Phoenix Suns',
            'Golden State Warriors', 'Dallas Mavericks'
        ]
        return equipo in equipos_alto_scoring
```

### 3. Integración con Meta-Modelo Existente

#### Modificaciones Propuestas
```python
# metamodel/integration_improvements.py

def mejorar_integracion_meta_modelo():
    """
    Función para integrar las mejoras al meta-modelo existente
    """
    
    # 1. Actualizar el meta_modelo.py principal
    codigo_actualizacion = '''
    # ===========================================
    # MEJORAS AL META-MODELO - NOVIEMBRE 2025
    # ===========================================
    
    # Importar nuevos componentes
    from .improved_outlier_detector import ImprovedOutlierDetector
    from .adaptive_reality_check import AdaptiveRealityCheck
    
    # En la clase MetaModeloInteligente, modificar __init__:
    def __init__(self, config_path: str = None, league_name: str = "NBA"):
        # ... código existente ...
        
        # Reemplazar detector de outliers
        self.detector_outliers = ImprovedOutlierDetector(league_name, self.config)
        
        # Agregar reality check adaptativo
        self.reality_check_adaptativo = AdaptiveRealityCheck(league_name, self.config)
    
    # Modificar método procesar_prediccion:
    def procesar_prediccion(
        self,
        prediccion_base: float,
        contexto_partido: Dict[str, Any],
        senales_live: Dict[str, Any] = None,
        q_scores: Dict[str, int] = None,
        quarter_stage: str = None,
        equipos: List[str] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Procesa predicción con mejoras integradas
        """
        # 1. Análisis de contexto (existente)
        contexto_analizado = self.analizador_contexto.analizar(
            prediccion_base, contexto_partido, senales_live, q_scores, quarter_stage
        )
        
        # 2. Detección mejorada de outliers
        outlier_info = self.detector_outliers.detectar_outlier_mejorado(
            prediccion_base, equipos or [], contexto_analizado, q_scores
        )
        
        # 3. Reality Check adaptativo
        prediccion_ajustada, rc_info = self.reality_check_adaptativo.aplicar_ajuste_adaptativo(
            prediccion_base, outlier_info, equipos or [], contexto_analizado, quarter_stage
        )
        
        # 4. Aplicar ajustes contextuales adicionales
        ajustes_contexto = self.analizador_contexto.obtener_ajustes_contexto(
            contexto_analizado, quarter_stage
        )
        
        for ajuste in ajustes_contexto:
            prediccion_ajustada *= ajuste['factor']
        
        # 5. Aplicar límites inteligentes
        prediccion_ajustada = self._aplicar_limites_inteligentes(
            prediccion_ajustada, contexto_analizado, outlier_info
        )
        
        # 6. Preparar información de ajuste
        info_ajuste = {
            'prediccion_base': prediccion_base,
            'prediccion_ajustada': prediccion_ajustada,
            'ajuste_total': prediccion_ajustada - prediccion_base,
            'outlier_info': outlier_info,
            'rc_info': rc_info,
            'ajustes_contexto': ajustes_contexto,
            'contexto_analizado': contexto_analizado,
            'timestamp': datetime.now().isoformat()
        }
        
        return prediccion_ajustada, info_ajuste
    '''
    
    return codigo_actualizacion
```

## 📊 Sistema de Validación y Testing

### 1. Script de Testing para Mejoras

```python
# tests/test_mejoras_rc.py

def test_mejoras_reality_check():
    """
    Suite de tests para validar las mejoras implementadas
    """
    
    # Casos de prueba basados en datos reales
    casos_prueba = [
        {
            'nombre': 'Miami vs NY - Outlier Legítimo',
            'equipos': ['Miami Heat', 'New York Knicks'],
            'prediccion_base': 270.5,
            'contexto': {
                'pace_actual': 108,
                'q2_spike': False,
                'intensity_drop': False
            },
            'resultado_esperado': {
                'es_outlier': True,
                'tipo': 'outlier_legitimo_alto_scoring',
                'aplicar_ajuste': False
            }
        },
        {
            'nombre': 'Boston vs LA Clippers - Sin Ajuste',
            'equipos': ['Boston Celtics', 'Los Angeles Clippers'],
            'prediccion_base': 222.5,
            'contexto': {
                'pace_actual': 102,
                'q2_spike': False,
                'intensity_drop': False
            },
            'resultado_esperado': {
                'es_outlier': False,
                'aplicar_ajuste': False
            }
        },
        {
            'nombre': 'Utah vs Atlanta - Optimismo Detectado',
            'equipos': ['Utah Jazz', 'Atlanta Hawks'],
            'prediccion_base': 285.5,
            'contexto': {
                'pace_actual': 95,
                'q2_spike': False,
                'intensity_drop': True
            },
            'resultado_esperado': {
                'es_outlier': True,
                'tipo': 'optimismo_sistemático',
                'aplicar_ajuste': True
            }
        }
    ]
    
    # Ejecutar tests
    resultados = []
    for caso in casos_prueba:
        resultado = ejecutar_caso_prueba(caso)
        resultados.append({
            'caso': caso['nombre'],
            'exitoso': resultado['exitoso'],
            'detalles': resultado['detalles']
        })
    
    return resultados

def ejecutar_caso_prueba(caso):
    """
    Ejecuta un caso de prueba específico
    """
    # Implementar lógica de test
    pass
```

### 2. Sistema de Monitoreo Continuo

```python
# metamodel/monitoring_system.py

class MonitoringSystem:
    """
    Sistema para monitorear el rendimiento de las mejoras
    """
    
    def __init__(self):
        self.metricas_diarias = []
        self.alertas = []
    
    def registrar_prediccion(
        self,
        prediccion_info: Dict[str, Any],
        resultado_real: float
    ):
        """
        Registra una predicción y su resultado para análisis
        """
        metricas = {
            'timestamp': datetime.now().isoformat(),
            'prediccion_base': prediccion_info['prediccion_base'],
            'prediccion_ajustada': prediccion_info['prediccion_ajustada'],
            'resultado_real': resultado_real,
            'error_base': abs(prediccion_info['prediccion_base'] - resultado_real),
            'error_ajustado': abs(prediccion_info['prediccion_ajustada'] - resultado_real),
            'mejora_rc': prediccion_info['error_base'] > prediccion_info['error_ajustado'],
            'outlier_detectado': prediccion_info['outlier_info']['es_outlier'],
            'tipo_outlier': prediccion_info['outlier_info'].get('tipo'),
            'rc_aplicado': prediccion_info['rc_info']['aplicado']
        }
        
        self.metricas_diarias.append(metricas)
        
        # Verificar alertas
        self._verificar_alertas(metricas)
    
    def _verificar_alertas(self, metricas: Dict[str, Any]):
        """
        Verifica si hay condiciones que requieren atención
        """
        # Alerta 1: RC aplicando muchos ajustes incorrectos
        if metricas['rc_aplicado'] and not metricas['mejora_rc']:
            self.alertas.append({
                'tipo': 'rc_ajuste_incorrecto',
                'timestamp': metricas['timestamp'],
                'detalles': f"Error ajustado: {metricas['error_ajustado']:.1f} vs Error base: {metricas['error_base']:.1f}"
            })
        
        # Alerta 2: Outliers no detectados correctamente
        if metricas['error_base'] > 20 and not metricas['outlier_detectado']:
            self.alertas.append({
                'tipo': 'outlier_no_detectado',
                'timestamp': metricas['timestamp'],
                'detalles': f"Error grande sin detección: {metricas['error_base']:.1f} pts"
            })
    
    def generar_reporte_diario(self) -> Dict[str, Any]:
        """
        Genera reporte diario de rendimiento
        """
        if not self.metricas_diarias:
            return {'error': 'Sin datos para generar reporte'}
        
        # Calcular métricas agregadas
        total_predicciones = len(self.metricas_diarias)
        predicciones_con_rc = sum(1 for m in self.metricas_diarias if m['rc_aplicado'])
        mejoras_rc = sum(1 for m in self.metricas_diarias if m['mejora_rc'])
        
        reporte = {
            'fecha': datetime.now().strftime('%Y-%m-%d'),
            'total_predicciones': total_predicciones,
            'predicciones_con_rc': predicciones_con_rc,
            'tasa_mejora_rc': (mejoras_rc / predicciones_con_rc * 100) if predicciones_con_rc > 0 else 0,
            'error_promedio_base': sum(m['error_base'] for m in self.metricas_diarias) / total_predicciones,
            'error_promedio_ajustado': sum(m['error_ajustado'] for m in self.metricas_diarias) / total_predicciones,
            'alertas_del_dia': len(self.alertas),
            'tipos_outlier_detectados': {
                m['tipo_outlier'] for m in self.metricas_diarias 
                if m['outlier_detectado'] and m['tipo_outlier']
            }
        }
        
        return reporte
```

## 📋 Plan de Implementación Detallado

### Semana 1 (Crítico)
- [ ] **Lunes**: Implementar ImprovedOutlierDetector
- [ ] **Martes**: Implementar AdaptiveRealityCheck
- [ ] **Miércoles**: Integrar con meta-modelo existente
- [ ] **Jueves**: Crear suite de tests
- [ ] **Viernes**: Validar con datos históricos

### Semana 2 (Validación)
- [ ] **Lunes**: Ejecutar tests completos
- [ ] **Martes**: Implementar sistema de monitoreo
- [ ] **Miércoles**: Análisis de resultados iniciales
- [ ] **Jueves**: Ajustes finos basados en resultados
- [ ] **Viernes**: Documentación y deploy

### Semana 3-4 (Optimización)
- [ ] Implementar sistema ensemble
- [ ] Agregar aprendizaje continuo
- [ ] Crear dashboard de monitoreo
- [ ] Validación extensiva con datos históricos

## 🎯 Métricas de Éxito

### Objetivos Inmediatos (Semana 1-2)
| Métrica | Actual | Objetivo |
|---------|--------|----------|
| **Detección de outliers legítimos** | 0% | 80% |
| **Reducción de falsos negativos RC** | 33.3% | < 15% |
| **Precisión en equipos alto-scoring** | 33.3% | 70% |

### Objetivos Mediano Plazo (Mes)
| Métrica | Actual | Objetivo |
|---------|--------|----------|
| **Tasa general de aciertos** | 58.3% | 75% |
| **Precisión en predicciones HT** | 55.6% | 75% |
| **Reducción de sesgo sistemático** | +6.5 pts | < +2 pts |

## 🚀 Plan de Rollout

### Fase 1: Testing Controlado
- Implementar en entorno de desarrollo
- Validar con datos históricos conocidos
- Tests A/B con predicciones recientes

### Fase 2: Producción Parcial
- Activar en 50% de las predicciones
- Monitorear métricas en tiempo real
- Ajustes basados en resultados iniciales

### Fase 3: Producción Completa
- Activar en 100% de las predicciones
- Monitoreo continuo 24/7
- Sistema de rollback automático si degrada rendimiento

---

*Plan de implementación creado: 17 de noviembre de 2025*
*Inicio de implementación: 18 de noviembre de 2025*
*Revisión inicial: 25 de noviembre de 2025*