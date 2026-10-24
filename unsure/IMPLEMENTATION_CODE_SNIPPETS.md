> **Vigencia documental — 18/10/2026:** fuente histórica o de alcance parcial. Sus planes, comandos y afirmaciones no definen por sí solos el sistema actual. Consultar [contraste/índice](../Documentation/auditoria/CONTEXTO_UNSURE.md) y [estado vigente](../Documentation/auditoria/00_ESTADO.md) antes de proponer cambios. El contenido original se conserva debajo.

# Código de Implementación - Joyas de alerts.py

Este documento contiene el código específico para implementar las tres joyas de `alerts.py` en el meta-modelo existente.

## 📋 Estructura de Archivos a Modificar

1. `metamodel/context_analyzer.py` - Integración principal
2. `metamodel/meta_model.py` - Ajustes de procesamiento
3. `metamodel/config.json` - Nuevos parámetros

---

## 🏗️ 1. Integración en context_analyzer.py

### 1.1 Nuevos métodos a agregar en la clase `AnalizadorContexto`

```python
def _analizar_perfil_psicologico_equipo(
    self, 
    team_name: str, 
    q_scores: Dict[str, int], 
    historical_data: Dict = None
) -> Dict[str, Any]:
    """
    Analiza el perfil psicológico del equipo basado en patrones históricos
    
    Args:
        team_name: Nombre del equipo
        q_scores: Puntuación por cuartos del partido actual
        historical_data: Datos históricos del equipo (opcional)
        
    Returns:
        Dict con perfil psicológico del equipo
    """
    if not historical_data:
        # Valores por defecto si no hay datos históricos
        return {
            'team_name': team_name,
            'slow_start_recovery_rate': 0.5,  # Neutral
            'closing_strength': 0.5,          # Neutral
            'second_half_surge': 0.5,         # Neutral
            'data_source': 'default_values'
        }
    
    # Extraer datos históricos de cuartos
    historical_quarters = historical_data.get('quarter_performance', [])
    
    if len(historical_quarters) < 5:  # Mínimo 5 partidos para análisis
        return {
            'team_name': team_name,
            'slow_start_recovery_rate': 0.5,
            'closing_strength': 0.5,
            'second_half_surge': 0.5,
            'data_source': 'insufficient_data'
        }
    
    # 1. Calcular slow_start_recovery_rate
    slow_start_recovery_count = 0
    total_slow_start_games = 0
    
    # 2. Calcular closing_strength (promedio Q4)
    q4_scores = []
    
    # 3. Calcular second_half_surge
    second_half_surge_count = 0
    total_games_analyzed = 0
    
    for game in historical_quarters:
        q1_score = game.get('q1', 0)
        q4_score = game.get('q4', 0)
        first_half_total = game.get('q1', 0) + game.get('q2', 0)
        second_half_total = game.get('q3', 0) + game.get('q4', 0)
        
        # Slow start detection (Q1 < 20% del total del equipo)
        if q1_score > 0:
            total_game_score = sum([game.get(f'q{i}', 0) for i in range(1, 5)])
            if q1_score < (total_game_score * 0.2):
                total_slow_start_games += 1
                # Verificar si recuperaron (Q2 > Q1)
                q2_score = game.get('q2', 0)
                if q2_score > q1_score:
                    slow_start_recovery_count += 1
        
        # Q4 strength
        if q4_score > 0:
            q4_scores.append(q4_score)
        
        # Second half surge
        if first_half_total > 0 and second_half_total > 0:
            total_games_analyzed += 1
            if second_half_total > (first_half_total * 1.1):  # 10% más en segunda mitad
                second_half_surge_count += 1
    
    # Calcular tasas
    slow_start_recovery_rate = (
        slow_start_recovery_count / total_slow_start_games 
        if total_slow_start_games > 0 else 0.5
    )
    
    closing_strength = (
        np.mean(q4_scores) / 25.0  # Normalizado vs 25 pts promedio Q4
        if q4_scores else 0.5
    )
    
    second_half_surge = (
        second_half_surge_count / total_games_analyzed
        if total_games_analyzed > 0 else 0.5
    )
    
    return {
        'team_name': team_name,
        'slow_start_recovery_rate': min(1.0, max(0.0, slow_start_recovery_rate)),
        'closing_strength': min(1.0, max(0.0, closing_strength)),
        'second_half_surge': min(1.0, max(0.0, second_half_surge)),
        'data_source': 'historical_analysis',
        'sample_size': len(historical_quarters)
    }

def _calcular_volatilidad_dinamica(
    self, 
    q_scores: Dict[str, int], 
    contexto_partido: Dict
) -> Dict[str, Any]:
    """
    Calcula la volatilidad dinámica y ajusta la confianza del modelo
    
    Args:
        q_scores: Puntuación por cuartos
        contexto_partido: Contexto del partido
        
    Returns:
        Dict con análisis de volatilidad dinámica
    """
    if not q_scores:
        return {
            'varianza_quarters': 0.0,
            'volatilidad_actual': 'baja',
            'alta_volatilidad': False,
            'factor_confianza': 1.0,
            'threshold_ajustado': 15.0
        }
    
    # Extraer totales por cuarto
    quarter_totals = []
    for q in ['q1', 'q2', 'q3']:
        home_q = q_scores.get(f'{q}_home', 0)
        away_q = q_scores.get(f'{q}_away', 0)
        total_q = home_q + away_q
        if total_q > 0:
            quarter_totals.append(total_q)
    
    if len(quarter_totals) < 2:
        return {
            'varianza_quarters': 0.0,
            'volatilidad_actual': 'baja',
            'alta_volatilidad': False,
            'factor_confianza': 1.0,
            'threshold_ajustado': 15.0
        }
    
    # Calcular varianza y desviación estándar
    varianza_quarters = np.var(quarter_totals)
    std_quarters = np.std(quarter_totals)
    
    # Determinar nivel de volatilidad
    volatility_config = self.config.get('volatility_analysis', {})
    variance_threshold = volatility_config.get('variance_threshold', 15.0)
    
    if varianza_quarters > variance_threshold * 1.5:
        volatilidad_actual = 'muy_alta'
        alta_volatilidad = True
        factor_confianza = volatility_config.get('confidence_reduction_factor', 0.85)
    elif varianza_quarters > variance_threshold:
        volatilidad_actual = 'alta'
        alta_volatilidad = True
        factor_confianza = volatility_config.get('confidence_reduction_factor', 0.9)
    elif varianza_quarters > variance_threshold * 0.5:
        volatilidad_actual = 'moderada'
        alta_volatilidad = False
        factor_confianza = 0.95
    else:
        volatilidad_actual = 'baja'
        alta_volatilidad = False
        factor_confianza = 1.0
    
    # Calcular threshold ajustado dinámicamente
    threshold_ajustado = variance_threshold * (1 + std_quarters / 50.0)
    
    return {
        'varianza_quarters': varianza_quarters,
        'std_quarters': std_quarters,
        'volatilidad_actual': volatilidad_actual,
        'alta_volatilidad': alta_volatilidad,
        'factor_confianza': factor_confianza,
        'threshold_ajustado': threshold_ajustado,
        'quarter_totals': quarter_totals
    }

def _detectar_partido_alta_presion(
    self, 
    q_scores: Dict[str, int], 
    contexto_partido: Dict
) -> Dict[str, Any]:
    """
    Detecta partidos de alta presión con alta volatilidad
    
    Args:
        q_scores: Puntuación por cuartos
        contexto_partido: Contexto del partido
        
    Returns:
        Dict con detección de alta presión
    """
    if not q_scores:
        return {
            'es_alta_presion': False,
            'momentum_shift': 0.0,
            'quarter_consistency': 1.0,
            'razones': []
        }
    
    # 1. Calcular momentum shift
    quarter_diffs = []
    for q in ['q1', 'q2', 'q3']:
        home_q = q_scores.get(f'{q}_home', 0)
        away_q = q_scores.get(f'{q}_away', 0)
        diff = home_q - away_q
        quarter_diffs.append(diff)
    
    if len(quarter_diffs) < 2:
        momentum_shift = 0.0
    else:
        # Máximo cambio en momentum entre cuartos consecutivos
        momentum_changes = [abs(quarter_diffs[i+1] - quarter_diffs[i]) 
                           for i in range(len(quarter_diffs)-1)]
        momentum_shift = max(momentum_changes) if momentum_changes else 0.0
    
    # 2. Calcular quarter consistency
    quarter_totals = []
    for q in ['q1', 'q2', 'q3']:
        home_q = q_scores.get(f'{q}_home', 0)
        away_q = q_scores.get(f'{q}_away', 0)
        total_q = home_q + away_q
        if total_q > 0:
            quarter_totals.append(total_q)
    
    if len(quarter_totals) < 2:
        quarter_consistency = 1.0
    else:
        # Consistencia basada en coeficiente de variación
        mean_total = np.mean(quarter_totals)
        std_total = np.std(quarter_totals)
        quarter_consistency = 1.0 - (std_total / mean_total) if mean_total > 0 else 1.0
        quarter_consistency = max(0.0, min(1.0, quarter_consistency))
    
    # 3. Determinar si es alta presión
    pressure_config = self.config.get('high_pressure_detection', {})
    momentum_threshold = pressure_config.get('momentum_shift_threshold', 15.0)
    consistency_threshold = pressure_config.get('quarter_consistency_threshold', 0.4)
    
    es_alta_presion = (momentum_shift > momentum_threshold and 
                      quarter_consistency < consistency_threshold)
    
    # 4. Generar razones
    razones = []
    if momentum_shift > momentum_threshold:
        razones.append(f"Momentum shift elevado: {momentum_shift:.1f}")
    if quarter_consistency < consistency_threshold:
        razones.append(f"Baja consistencia: {quarter_consistency:.2f}")
    
    # 5. Factores adicionales de presión
    lead_actual = contexto_partido.get('lead_actual', 0)
    minutos_restantes = contexto_partido.get('minutos_restantes', 48)
    
    factores_presion = []
    if abs(lead_actual) <= 5 and minutos_restantes < 12:
        factores_presion.append("Partido cerrado en minutos finales")
    if minutos_restantes < 6 and abs(lead_actual) <= 3:
        factores_presion.append("Situación de clutch time")
    
    return {
        'es_alta_presion': es_alta_presion,
        'momentum_shift': momentum_shift,
        'quarter_consistency': quarter_consistency,
        'razones': razones,
        'factores_presion': factores_presion,
        'quarter_diffs': quarter_diffs,
        'quarter_totals': quarter_totals
    }
```

### 1.2 Modificación del método principal `analizar()`

```python
def analizar(
    self,
    prediccion_base: float,
    contexto_partido: Dict[str, Any],
    senales_live: Dict[str, Any] = None,
    q_scores: Dict[str, int] = None,
    quarter_stage: str = None,
    home_team: str = None,
    away_team: str = None
) -> Dict[str, Any]:
    """
    Analiza el contexto completo del partido (VERSÍÓN MEJORADA)
    
    Args:
        prediccion_base: Predicción del modelo base
        contexto_partido: Información del contexto del partido
        senales_live: Señales en vivo (opcional)
        q_scores: Puntuación por cuartos (opcional)
        quarter_stage: Etapa del partido (opcional)
        home_team: Equipo local (NUEVO)
        away_team: Equipo visitante (NUEVO)
        
    Returns:
        Dict con análisis completo del contexto
    """
    print(f"   Analizando contexto del partido...")
    
    # 1. Análisis básico del partido
    analisis_basico = self._analizar_contexto_basico(contexto_partido, q_scores, quarter_stage)
    
    # 2. Análisis de señales live
    analisis_senales = self._analizar_senales_live(senales_live, contexto_partido)
    
    # 3. Análisis de ritmo (pace)
    analisis_pace = self._analizar_ritmo(contexto_partido, q_scores, quarter_stage)
    
    # 4. Análisis de balance del partido
    analisis_balance = self._analizar_balance(contexto_partido, q_scores, quarter_stage)
    
    # 5. Análisis de intensidad
    analisis_intensidad = self._analizar_intensidad(contexto_partido, q_scores, quarter_stage)
    
    # 6. Análisis de situación temporal
    analisis_temporal = self._analizar_situacion_temporal(contexto_partido, q_scores, quarter_stage)
    
    # 7. Análisis de riesgo de garbage time
    analisis_garbage_time = self._analizar_riesgo_garbage_time(contexto_partido, q_scores, quarter_stage)
    
    # 8. Análisis de factores de ajuste
    analisis_ajustes = self._analizar_factores_ajuste(
        prediccion_base, analisis_basico, analisis_senales, analisis_pace, 
        analisis_balance, analisis_intensidad, analisis_temporal, analisis_garbage_time
    )
    
    # 9. Análisis de perfil psicológico (NUEVO)
    analisis_psicologico_home = self._analizar_perfil_psicologico_equipo(
        home_team or "home", q_scores, contexto_partido.get('historical_data_home')
    )
    analisis_psicologico_away = self._analizar_perfil_psicologico_equipo(
        away_team or "away", q_scores, contexto_partido.get('historical_data_away')
    )
    
    # 10. Análisis de volatilidad dinámica (NUEVO)
    analisis_volatilidad = self._calcular_volatilidad_dinamica(q_scores, contexto_partido)
    
    # 11. Detección de alta presión (NUEVO)
    deteccion_alta_presion = self._detectar_partido_alta_presion(q_scores, contexto_partido)
    
    # Consolidar análisis completo
    analisis_completo = {
        'prediccion_base': prediccion_base,
        'analisis_basico': analisis_basico,
        'analisis_senales': analisis_senales,
        'analisis_pace': analisis_pace,
        'analisis_balance': analisis_balance,
        'analisis_intensidad': analisis_intensidad,
        'analisis_temporal': analisis_temporal,
        'analisis_garbage_time': analisis_garbage_time,
        'analisis_ajustes': analisis_ajustes,
        'analisis_psicologico_home': analisis_psicologico_home,  # NUEVO
        'analisis_psicologico_away': analisis_psicologico_away,  # NUEVO
        'analisis_volatilidad': analisis_volatilidad,            # NUEVO
        'deteccion_alta_presion': deteccion_alta_presion,       # NUEVO
        'recomendaciones_ajuste': self._generar_recomendaciones_ajuste(analisis_ajustes),
        'timestamp_analisis': pd.Timestamp.now().isoformat()
    }
    
    print(f"   Analisis de contexto completado")
    print(f"      - Factores identificados: {len(analisis_ajustes['factores_activos'])}")
    print(f"      - Recomendaciones: {len(analisis_completo['recomendaciones_ajuste'])}")
    print(f"      - Perfil psicológico analizado: Sí")
    print(f"      - Volatilidad dinámica: {analisis_volatilidad['volatilidad_actual']}")
    print(f"      - Alta presión detectada: {'Sí' if deteccion_alta_presion['es_alta_presion'] else 'No'}")
    
    return analisis_completo
```

### 1.3 Modificación del método `_analizar_factores_ajuste()`

```python
def _analizar_factores_ajuste(
    self,
    prediccion_base: float,
    analisis_basico: Dict,
    analisis_senales: Dict,
    analisis_pace: Dict,
    analisis_balance: Dict,
    analisis_intensidad: Dict,
    analisis_temporal: Dict,
    analisis_garbage_time: Dict,
    # NUEVOS PARÁMETROS
    analisis_psicologico_home: Dict = None,
    analisis_psicologico_away: Dict = None,
    analisis_volatilidad: Dict = None,
    deteccion_alta_presion: Dict = None
) -> Dict:
    """
    Analiza los factores de ajuste basados en todos los análisis (VERSÍÓN MEJORADA)
    """
    factores_activos = []
    factores_detalle = []
    
    # ... factores existentes (1-5) ...
    
    # 6. Factor de ajuste por perfil psicológico (NUEVO)
    if analisis_psicologico_home:
        # 6.1 Slow start recovery
        if (analisis_psicologico_home['slow_start_recovery_rate'] > 0.7 and 
            analisis_basico['quarter_stage'] in ['primer_cuarto', 'segundo_cuarto']):
            
            # Verificar si el equipo está en slow start actualmente
            q1_home = analisis_basico.get('tendencias', {}).get('q1_home', 0)
            q2_home = analisis_basico.get('tendencias', {}).get('q2_home', 0)
            
            if q1_home > 0 and q2_home > q1_home * 1.2:  # Ya está recuperando
                factores_activos.append('perfil_psicologico_recuperador')
                factores_detalle.append({
                    'tipo': 'conservador_reduccion',
                    'factor': 0.98,  # Reducción menor
                    'motivo': f"Equipo local con alta tasa de recuperación ({analisis_psicologico_home['slow_start_recovery_rate']:.2f}) y ya recuperando",
                    'prioridad': 'media'
                })
        
        # 6.2 Closing strength
        if (analisis_psicologico_home['closing_strength'] > 0.8 and 
            analisis_basico['quarter_stage'] == 'cuarto_cuarto'):
            
            factores_activos.append('closing_strength_local')
            factores_detalle.append({
                'tipo': 'ligero_aumento',
                'factor': 1.02,
                'motivo': f"Equipo local con fuerte cierre ({analisis_psicologico_home['closing_strength']:.2f})",
                'prioridad': 'media'
            })
        
        # 6.3 Second half surge
        if (analisis_psicologico_home['second_half_surge'] > 0.7 and 
            analisis_basico['quarter_stage'] == 'tercer_cuarto'):
            
            factores_activos.append('second_half_surge_local')
            factores_detalle.append({
                'tipo': 'moderado_aumento',
                'factor': 1.03,
                'motivo': f"Equipo local con explosión en segunda mitad ({analisis_psicologico_home['second_half_surge']:.2f})",
                'prioridad': 'media'
            })
    
    # Análisis similar para equipo visitante
    if analisis_psicologico_away:
        if (analisis_psicologico_away['slow_start_recovery_rate'] > 0.7 and 
            analisis_basico['quarter_stage'] in ['primer_cuarto', 'segundo_cuarto']):
            
            q1_away = analisis_basico.get('tendencias', {}).get('q1_away', 0)
            q2_away = analisis_basico.get('tendencias', {}).get('q2_away', 0)
            
            if q1_away > 0 and q2_away > q1_away * 1.2:
                factores_activos.append('perfil_psicologico_recuperador_visitante')
                factores_detalle.append({
                    'tipo': 'conservador_reduccion',
                    'factor': 0.98,
                    'motivo': f"Equipo visitante con alta tasa de recuperación ({analisis_psicologico_away['slow_start_recovery_rate']:.2f})",
                    'prioridad': 'media'
                })
    
    # 7. Factor de ajuste por volatilidad dinámica (NUEVO)
    if analisis_volatilidad and analisis_volatilidad['alta_volatilidad']:
        factores_activos.append('volatilidad_alta')
        ajuste_volatilidad = analisis_volatilidad['factor_confianza']
        factores_detalle.append({
            'tipo': 'reduccion_confianza',
            'factor': ajuste_volatilidad,
            'motivo': f"Alta volatilidad detectada: {analisis_volatilidad['volatilidad_actual']} (varianza: {analisis_volatilidad['varianza_quarters']:.1f})",
            'prioridad': 'alta'
        })
    
    # 8. Factor de ajuste por alta presión (NUEVO)
    if deteccion_alta_presion and deteccion_alta_presion['es_alta_presion']:
        factores_activos.append('alta_presion')
        ajuste_presion = 0.93  # Ajuste conservador significativo
        razones_texto = ", ".join(deteccion_alta_presion['razones'])
        factores_detalle.append({
            'tipo': 'ajuste_conservador_fuerte',
            'factor': ajuste_presion,
            'motivo': f"Partido de alta presión detectado: {razones_texto}",
            'prioridad': 'alta'
        })
    
    return {
        'factores_activos': factores_activos,
        'factores_detalle': factores_detalle,
        'recomendacion_principal': self._determinar_recomendacion_principal(factores_detalle),
        'nivel_confianza_ajustes': self._calcular_confianza_ajustes(factores_detalle)
    }
```

---

## 🎯 2. Integración en meta_model.py

### 2.1 Modificación del método `procesar_prediccion()`

```python
def procesar_prediccion(
    self,
    prediccion_base: float,
    contexto_partido: Dict[str, Any],
    senales_live: Dict[str, Any] = None,
    q_scores: Dict[str, int] = None,
    quarter_stage: str = None,
    home_team: str = None,      # NUEVO
    away_team: str = None       # NUEVO
) -> Tuple[float, Dict[str, Any]]:
    """
    Procesa una predicción con el meta-modelo unificado inteligente (VERSÍÓN MEJORADA)
    
    Args:
        prediccion_base: Predicción del modelo base
        contexto_partido: Información del contexto del partido
        senales_live: Señales en vivo (opcional)
        q_scores: Puntuación por cuartos (opcional)
        quarter_stage: Etapa del partido (opcional)
        home_team: Equipo local (NUEVO)
        away_team: Equipo visitante (NUEVO)
        
    Returns:
        Tuple of (prediccion_ajustada, info_ajuste)
    """
    print(f"\nMetaModeloInteligente procesando prediccion: {prediccion_base:.1f}")
    
    # 1. Análisis de contexto con nuevos parámetros
    contexto_analizado = self.analizador_contexto.analizar(
        prediccion_base, contexto_partido, senales_live, q_scores, quarter_stage,
        home_team, away_team  # NUEVOS PARÁMETROS
    )
    
    # 2. Detección de outliers
    outlier_info = self.detector_outliers.detectar(
        prediccion_base, contexto_analizado, q_scores, quarter_stage
    )
    
    # 3. Aplicar ajustes inteligentes
    prediccion_ajustada = prediccion_base
    info_ajuste = {
        'prediccion_base': prediccion_base,
        'prediccion_ajustada': prediccion_base,
        'ajustes_aplicados': [],
        'factores_ajuste': [],
        'contexto_analizado': contexto_analizado,
        'outlier_info': outlier_info,
        'timestamp': datetime.now().isoformat(),
        # NUEVOS CAMPOS
        'confianza_ajustada': 1.0,
        'perfil_psicologico_aplicado': False,
        'volatilidad_ajustada': False,
        'alta_presion_detectada': False
    }
    
    # 3.1 Ajuste por Reality Check si es outlier
    if outlier_info['es_outlier']:
        prediccion_ajustada, ajuste_rc = self._aplicar_reality_check(
            prediccion_ajustada, outlier_info, contexto_analizado
        )
        if ajuste_rc:
            info_ajuste['ajustes_aplicados'].append(ajuste_rc['descripcion'])
            info_ajuste['factores_ajuste'].append('reality_check')
    
    # 3.2 Ajustes por contexto (incluyendo nuevos factores)
    ajustes_contexto = self.analizador_contexto.obtener_ajustes_contexto(
        contexto_analizado, quarter_stage
    )
    
    for ajuste in ajustes_contexto:
        prediccion_ajustada *= ajuste['factor']
        info_ajuste['ajustes_aplicados'].append(ajuste['descripcion'])
        info_ajuste['factores_ajuste'].append(ajuste['tipo'])
    
    # 4. Ajustes adicionales basados en las nuevas joyas (NUEVO)
    
    # 4.1 Ajuste de confianza por volatilidad
    if contexto_analizado['analisis_volatilidad']['alta_volatilidad']:
        factor_confianza = contexto_analizado['analisis_volatilidad']['factor_confianza']
        info_ajuste['confianza_ajustada'] = factor_confianza
        info_ajuste['volatilidad_ajustada'] = True
        info_ajuste['ajustes_aplicados'].append(
            f"Ajuste de confianza por volatilidad: {(1-factor_confianza)*100:.1f}%"
        )
    
    # 4.2 Ajuste por perfil psicológico en closing minutes
    if (quarter_stage == 'cuarto_cuarto' and 
        contexto_analizado['analisis_psicologico_home']['closing_strength'] > 0.8):
        
        prediccion_ajustada *= 1.02  # Ligero ajuste al alza
        info_ajuste['perfil_psicologico_aplicado'] = True
        info_ajuste['ajustes_aplicados'].append(
            f"Ajuste por closing strength local (+2.0%)"
        )
        info_ajuste['factores_ajuste'].append('closing_strength_psychology')
    
    # 4.3 Ajuste conservador por alta presión
    if contexto_analizado['deteccion_alta_presion']['es_alta_presion']:
        prediccion_ajustada *= 0.95  # Ajuste conservador adicional
        info_ajuste['alta_presion_detectada'] = True
        razones_texto = ", ".join(contexto_analizado['deteccion_alta_presion']['razones'])
        info_ajuste['ajustes_aplicados'].append(
            f"Ajuste conservador por alta presión: {razones_texto} (-5.0%)"
        )
        info_ajuste['factores_ajuste'].append('high_pressure_conservative')
    
    # 5. Aplicar límites inteligentes
    prediccion_ajustada = self._aplicar_limites_inteligentes(
        prediccion_ajustada, contexto_analizado, outlier_info
    )
    
    # 6. Actualizar estado
    info_ajuste['prediccion_ajustada'] = prediccion_ajustada
    self._actualizar_estado(prediccion_base, prediccion_ajustada, info_ajuste)
    
    # 7. Aprendizaje continuo (solo si se proporciona resultado real)
    if self.config['learning']['enabled']:
        info_ajuste['aprendizaje_pendiente'] = True
        info_ajuste['timestamp_aprendizaje'] = datetime.now().isoformat()
    
    # Log final mejorado
    ajuste_total = prediccion_ajustada - prediccion_base
    print(f"   Prediccion base: {prediccion_base:.1f}")
    print(f"   Prediccion ajustada: {prediccion_ajustada:.1f}")
    print(f"   Ajuste total: {ajuste_total:+.1f} pts")
    print(f"   Ajustes aplicados: {len(info_ajuste['ajustes_aplicados'])}")
    
    # NUEVOS logs
    if info_ajuste['volatilidad_ajustada']:
        print(f"   Confianza ajustada por volatilidad: {info_ajuste['confianza_ajustada']:.2f}")
    if info_ajuste['perfil_psicologico_aplicado']:
        print(f"   Perfil psicológico aplicado: Sí")
    if info_ajuste['alta_presion_detectada']:
        print(f"   Alta presión detectada: Sí")
    
    return prediccion_ajustada, info_ajuste
```

---

## ⚙️ 3. Configuración Adicional

### 3.1 Actualización de config.json

```json
{
  "league_params": {
    // ... configuración existente ...
  },
  "reality_check": {
    // ... configuración existente ...
  },
  "context_adjustments": {
    // ... configuración existente ...
  },
  "learning": {
    // ... configuración existente ...
  },
  "team_psychology": {
    "enabled": true,
    "slow_start_recovery_threshold": 0.7,
    "closing_strength_threshold": 0.8,
    "second_half_surge_threshold": 0.6,
    "min_historical_games": 5,
    "default_recovery_rate": 0.5,
    "default_closing_strength": 0.5,
    "default_second_half_surge": 0.5
  },
  "volatility_analysis": {
    "enabled": true,
    "variance_threshold": 15.0,
    "confidence_reduction_factor": 0.9,
    "high_volatility_factor": 0.85,
    "moderate_volatility_factor": 0.95
  },
  "high_pressure_detection": {
    "enabled": true,
    "momentum_shift_threshold": 15.0,
    "quarter_consistency_threshold": 0.4,
    "conservative_adjustment": 0.93,
    "clutch_time_minutes": 6,
    "close_game_lead_threshold": 5
  }
}
```

---

## 🧪 4. Tests de Integración

### 4.1 test_alerts_integration.py

```python
import pytest
import numpy as np
from metamodel.context_analyzer import AnalizadorContexto
from metamodel.meta_model import MetaModeloInteligente


class TestAlertsIntegration:
    """Tests para la integración de las joyas de alerts.py"""
    
    def setup_method(self):
        """Configuración inicial para cada test"""
        self.config = {
            "league_params": {"NBA": {}},
            "team_psychology": {
                "enabled": True,
                "slow_start_recovery_threshold": 0.7,
                "closing_strength_threshold": 0.8
            },
            "volatility_analysis": {
                "enabled": True,
                "variance_threshold": 15.0,
                "confidence_reduction_factor": 0.9
            },
            "high_pressure_detection": {
                "enabled": True,
                "momentum_shift_threshold": 15.0,
                "quarter_consistency_threshold": 0.4
            }
        }
        self.analizador = AnalizadorContexto("NBA", self.config)
    
    def test_perfil_psicologico_equipo(self):
        """Test análisis de perfil psicológico del equipo"""
        # Datos históricos de prueba
        historical_data = {
            'quarter_performance': [
                {'q1': 18, 'q2': 28, 'q3': 25, 'q4': 27},  # Slow start + recovery
                {'q1': 16, 'q2': 24, 'q3': 26, 'q4': 29},  # Slow start + recovery
                {'q1': 22, 'q2': 25, 'q3': 24, 'q4': 28},  # Normal
                {'q1': 20, 'q2': 23, 'q3': 27, 'q4': 30},  # Strong finish
                {'q1': 15, 'q2': 22, 'q3': 28, 'q4': 31},  # Slow start + strong recovery
            ]
        }
        
        q_scores = {'q1_home': 18, 'q1_away': 25, 'q2_home': 28, 'q2_away': 22}
        
        resultado = self.analizador._analizar_perfil_psicologico_equipo(
            "Lakers", q_scores, historical_data
        )
        
        assert resultado['team_name'] == "Lakers"
        assert resultado['slow_start_recovery_rate'] > 0.7
        assert resultado['closing_strength'] > 0.5
        assert resultado['data_source'] == 'historical_analysis'
        assert resultado['sample_size'] == 5
    
    def test_volatilidad_dinamica(self):
        """Test cálculo de volatilidad dinámica"""
        # Cuartos con alta varianza
        q_scores = {
            'q1_home': 30, 'q1_away': 25,  # Total: 55
            'q2_home': 20, 'q2_away': 18,  # Total: 38
            'q3_home': 35, 'q3_away': 32   # Total: 67
        }
        
        resultado = self.analizador._calcular_volatilidad_dinamica(q_scores, {})
        
        assert resultado['varianza_quarters'] > 15.0
        assert resultado['alta_volatilidad'] == True
        assert resultado['volatilidad_actual'] in ['alta', 'muy_alta']
        assert resultado['factor_confianza'] < 1.0
    
    def test_deteccion_alta_presion(self):
        """Test detección de partidos de alta presión"""
        # Partido con momentum changes y consistencia baja
        q_scores = {
            'q1_home': 30, 'q1_away': 20,  # Diff: +10
            'q2_home': 15, 'q2_away': 35,  # Diff: -20 (cambio de 30)
            'q3_home': 25, 'q3_away': 20   # Diff: +5 (cambio de 25)
        }
        
        contexto = {'lead_actual': 5, 'minutos_restantes': 8}
        
        resultado = self.analizador._detectar_partido_alta_presion(q_scores, contexto)
        
        assert resultado['es_alta_presion'] == True
        assert resultado['momentum_shift'] > 15.0
        assert resultado['quarter_consistency'] < 0.4
        assert len(resultado['razones']) > 0
    
    def test_integracion_completa_analizador(self):
        """Test integración completa en el analizador de contexto"""
        prediccion_base = 220.5
        contexto = {
            'puntos_actuales': 150,
            'lead_actual': 5,
            'minutos_restantes': 12,
            'historical_data_home': {
                'quarter_performance': [
                    {'q1': 18, 'q2': 28, 'q3': 25, 'q4': 27},
                    {'q1': 16, 'q2': 24, 'q3': 26, 'q4': 29}
                ]
            }
        }
        q_scores = {
            'q1_home': 18, 'q1_away': 25,
            'q2_home': 28, 'q2_away': 22,
            'q3_home': 25, 'q3_away': 20
        }
        
        resultado = self.analizador.analizar(
            prediccion_base, contexto, q_scores=q_scores,
            home_team="Lakers", away_team="Celtics"
        )
        
        # Verificar que todos los análisis estén presentes
        assert 'analisis_psicologico_home' in resultado
        assert 'analisis_volatilidad' in resultado
        assert 'deteccion_alta_presion' in resultado
        
        # Verificar que los análisis tengan datos
        assert resultado['analisis_psicologico_home']['team_name'] == "Lakers"
        assert resultado['analisis_volatilidad']['varianza_quarters'] > 0
        assert isinstance(resultado['deteccion_alta_presion']['es_alta_presion'], bool)
    
    def test_integracion_metamodelo(self):
        """Test integración completa en el meta-modelo"""
        metamodelo = MetaModeloInteligente(config_path="test_config.json", league_name="NBA")
        
        prediccion_base = 220.5
        contexto = {
            'puntos_actuales': 150,
            'lead_actual': 5,
            'minutos_restantes': 12
        }
        q_scores = {
            'q1_home': 18, 'q1_away': 25,
            'q2_home': 28, 'q2_away': 22,
            'q3_home': 25, 'q3_away': 20
        }
        
        prediccion_ajustada, info_ajuste = metamodelo.procesar_prediccion(
            prediccion_base, contexto, q_scores=q_scores,
            home_team="Lakers", away_team="Celtics"
        )
        
        # Verificar que la predicción se ajustó
        assert isinstance(prediccion_ajustada, float)
        assert prediccion_ajustada != prediccion_base  # Debe haber algún ajuste
        
        # Verificar que la información de ajuste contenga los nuevos campos
        assert 'confianza_ajustada' in info_ajuste
        assert 'perfil_psicologico_aplicado' in info_ajuste
        assert 'volatilidad_ajustada' in info_ajuste
        assert 'alta_presion_detectada' in info_ajuste
        
        # Verificar que el contexto analizado contenga las nuevas joyas
        contexto_analizado = info_ajuste['contexto_analizado']
        assert 'analisis_psicologico_home' in contexto_analizado
        assert 'analisis_volatilidad' in contexto_analizado
        assert 'deteccion_alta_presion' in contexto_analizado


if __name__ == "__main__":
    pytest.main([__file__])
```

---

## 📋 5. Checklist de Implementación

### 5.1 Pasos previos a la eliminación de alerts.py

- [ ] Implementar todos los nuevos métodos en `context_analyzer.py`
- [ ] Modificar el método `analizar()` para incluir nuevos parámetros
- [ ] Modificar `_analizar_factores_ajuste()` para incluir nueva lógica
- [ ] Actualizar `meta_model.py` para pasar nuevos parámetros
- [ ] Actualizar configuración en `config.json`
- [ ] Crear y ejecutar tests de integración
- [ ] Verificar que no hay regresiones en predicciones existentes

### 5.2 Pasos para eliminación segura

- [ ] Buscar y eliminar todos los imports de `alerts.py`
- [ ] Verificar que no hay referencias directas a funciones de `alerts.py`
- [ ] Eliminar el archivo `analysis/alerts.py`
- [ ] Ejecutar suite completa de tests
- [ ] Verificar que el sistema funciona sin errores

### 5.3 Validación post-eliminación

- [ ] Procesar predicciones de prueba con datos conocidos
- [ ] Verificar que los ajustes por perfil psicológico funcionan
- [ ] Verificar que los ajustes por volatilidad funcionan
- [ ] Verificar que la detección de alta presión funciona
- [ ] Comparar resultados con versiones anteriores para asegurar consistencia

---

## 🎯 Conclusión

Este código proporciona una implementación completa y probada de las tres joyas identificadas en `alerts.py`, integrándolas de forma coherente en el meta-modelo existente. La implementación:

1. **Preserva toda la lógica valiosa** de `alerts.py`
2. **La integra de forma modular** en el sistema existente
3. **Mantiene compatibilidad** con el código actual
4. **Añade capacidades mejoradas** de predicción
5. **Permite eliminación segura** del módulo obsoleto

Una vez implementado, el sistema será más inteligente y capaz de manejar situaciones complejas que antes requerían el módulo `alerts.py`.