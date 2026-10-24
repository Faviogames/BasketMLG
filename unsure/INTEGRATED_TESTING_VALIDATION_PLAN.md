> **Vigencia documental — 18/10/2026:** fuente histórica o de alcance parcial. Sus planes, comandos y afirmaciones no definen por sí solos el sistema actual. Consultar [contraste/índice](../Documentation/auditoria/CONTEXTO_UNSURE.md) y [estado vigente](../Documentation/auditoria/00_ESTADO.md) antes de proponer cambios. El contenido original se conserva debajo.

# Plan de Pruebas y Validación Integradas

## Resumen

Este documento describe el plan completo de pruebas y validación para las mejoras implementadas en el sistema BasketMLG, incluyendo el bug corregido en Adaptive Split Detector, las mejoras en Run Detector y el mecanismo de decay en el Sistema de Aprendizaje.

## Objetivos de Validación

1. **Verificar corrección del bug** en Adaptive Split Detector
2. **Validar mejoras en Run Detector** con ContextAnalyzer
3. **Confirmar funcionamiento del decay** en Sistema de Aprendizaje
4. **Evaluar impacto integrado** en precisión de predicciones
5. **Asegurar estabilidad del sistema** bajo diversas condiciones

## Estrategia de Pruebas

### 1. Pruebas Unitarias

#### 1.1 Adaptive Split Detector

**Archivo**: `test_adaptive_split_detector.py`

```python
import unittest
from core.live_features import calculate_adaptive_split_detector

class TestAdaptiveSplitDetector(unittest.TestCase):
    
    def test_q3_early_minutes(self):
        """Test: PPM correcto con Q3 en progreso (minuto 2)"""
        q_scores = {
            'q1_home': 25, 'q1_away': 23,
            'q2_home': 28, 'q2_away': 26,
            'q3_home': 8, 'q3_away': 7  # 15 puntos en 2 min
        }
        
        time_info = {
            'time_into_quarter': 120,  # 2 minutos en segundos
            'minutes_played': 26.0
        }
        
        result = calculate_adaptive_split_detector(
            q_scores, 'q3_progress', time_info=time_info
        )
        
        # PPM debe ser 15/2 = 7.5 puntos por minuto, no 15/12 = 1.25
        self.assertGreater(result, 0.5)  # Debe detectar cambio significativo
    
    def test_q3_late_minutes(self):
        """Test: PPM correcto con Q3 avanzado (minuto 10)"""
        q_scores = {
            'q1_home': 25, 'q1_away': 23,
            'q2_home': 28, 'q2_away': 26,
            'q3_home': 35, 'q3_away': 33  # 68 puntos en 10 min
        }
        
        time_info = {
            'time_into_quarter': 600,  # 10 minutos en segundos
            'minutes_played': 34.0
        }
        
        result = calculate_adaptive_split_detector(
            q_scores, 'q3_progress', time_info=time_info
        )
        
        # PPM debe ser 68/10 = 6.8 puntos por minuto
        self.assertGreater(result, 0.5)  # Debe detectar cambio significativo
    
    def test_q3_insufficient_points(self):
        """Test: No activar detector con pocos puntos en Q3"""
        q_scores = {
            'q1_home': 25, 'q1_away': 23,
            'q2_home': 28, 'q2_away': 26,
            'q3_home': 4, 'q3_away': 3  # 7 puntos totales
        }
        
        time_info = {
            'time_into_quarter': 180,  # 3 minutos
            'minutes_played': 27.0
        }
        
        result = calculate_adaptive_split_detector(
            q_scores, 'q3_progress', time_info=time_info
        )
        
        # No debe activar con menos de 10 puntos
        self.assertEqual(result, 0.0)

if __name__ == '__main__':
    unittest.main()
```

#### 1.2 Run Detector con ContextAnalyzer

**Archivo**: `test_run_detector_context.py`

```python
import unittest
from unittest.mock import Mock
from core.live_api import LiveProcessor

class TestRunDetectorContext(unittest.TestCase):
    
    def setUp(self):
        """Configurar mock de ContextAnalyzer"""
        self.mock_context_analyzer = Mock()
        self.mock_context_analyzer.analizar.return_value = {
            'analisis_pace': {'es_pace_alto': True},
            'analisis_senales': {'impacto_total': 4},
            'analisis_garbage_time': {'activar_garbage_time': False},
            'analisis_balance': {'es_desbalanceado': False}
        }
    
    def test_run_detection_high_pace(self):
        """Test: Detección ajustada con pace alto"""
        q_scores = {
            'q1_home': 25, 'q1_away': 23,
            'q2_home': 28, 'q2_away': 26,
            'q3_home': 30, 'q3_away': 28
        }
        
        game_info = {'status': 'Q3 - 6:00'}
        
        # Simular último estado
        processor = LiveProcessor({})
        processor.last_q_scores = {
            'q1_home': 25, 'q1_away': 23,
            'q2_home': 28, 'q2_away': 26,
            'q3_home': 20, 'q3_away': 18
        }
        
        result = processor._calculate_run_detector(
            q_scores, game_info, self.mock_context_analyzer
        )
        
        # Con pace alto, el umbral debe ser más exigente
        self.assertTrue(result['context_used'])
        self.assertGreater(result['adaptive_thresholds']['run_threshold_base'], 6.0)
    
    def test_run_detection_garbage_time(self):
        """Test: Detección reducida con garbage time"""
        # Configurar mock para garbage time
        self.mock_context_analyzer.analizar.return_value = {
            'analisis_garbage_time': {'activar_garbage_time': True},
            'analisis_pace': {'es_pace_alto': False},
            'analisis_senales': {'impacto_total': 2},
            'analisis_balance': {'es_desbalanceado': False}
        }
        
        q_scores = {
            'q1_home': 25, 'q1_away': 23,
            'q2_home': 28, 'q2_away': 26,
            'q3_home': 30, 'q3_away': 28
        }
        
        game_info = {'status': 'Q4 - 2:00'}
        
        processor = LiveProcessor({})
        processor.last_q_scores = {
            'q1_home': 25, 'q1_away': 23,
            'q2_home': 28, 'q2_away': 26,
            'q3_home': 20, 'q3_away': 18
        }
        
        result = processor._calculate_run_detector(
            q_scores, game_info, self.mock_context_analyzer
        )
        
        # Con garbage time, el umbral debe ser mucho más exigente
        self.assertGreater(result['adaptive_thresholds']['run_threshold_base'], 9.0)
        self.assertLess(result['adaptive_thresholds']['strength_multiplier'], 1.0)

if __name__ == '__main__':
    unittest.main()
```

#### 1.3 Sistema de Aprendizaje con Decay

**Archivo**: `test_learning_system_decay.py`

```python
import unittest
import json
import tempfile
import os
from metamodel.learning_system import SistemaAprendizaje

class TestLearningSystemDecay(unittest.TestCase):
    
    def setUp(self):
        """Configurar sistema de aprendizaje con decay"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'config.json')
        
        # Configuración con decay activado
        config = {
            "learning": {
                "enabled": True,
                "factor_aprendizaje": 0.1,
                "decay": {
                    "decay_enabled": True,
                    "decay_rate": 0.95,
                    "valores_por_defecto": {
                        "sensibilidad_primeros_cuartos": 1.0,
                        "sensibilidad_tercer_cuarto": 1.0,
                        "conservadurismo_q4": 1.0
                    }
                }
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config, f)
        
        self.learning_system = SistemaAprendizaje('NBA', self.config_path)
    
    def test_decay_application(self):
        """Test: Aplicación correcta de decay"""
        # Establecer parámetro alejado del defecto
        self.learning_system.parametros_aprendidos['sensibilidad_primeros_cuartos'] = 1.5
        
        # Aplicar decay
        decay_config = self.learning_system.learning_config.get('decay', {})
        self.learning_system._aplicar_decay_parametros(decay_config)
        
        # Verificar que se aplicó decay hacia valor por defecto (1.0)
        valor_actual = self.learning_system.parametros_aprendidos['sensibilidad_primeros_cuartos']
        self.assertLess(valor_actual, 1.5)  # Debe reducirse
        self.assertGreater(valor_actual, 1.0)  # Pero no llegar al defecto inmediatamente
    
    def test_parameter_limits(self):
        """Test: Límites estrictos en parámetros"""
        # Establecer parámetro fuera de rango
        self.learning_system.parametros_aprendidos['sensibilidad_primeros_cuartos'] = 5.0
        
        # Validar y corregir
        self.learning_system._validar_y_corregir_parametros()
        
        # Verificar que se corrigió al rango permitido (0.5-2.0)
        valor_corregido = self.learning_system.parametros_aprendidos['sensibilidad_primeros_cuartos']
        self.assertLessEqual(valor_corregido, 2.0)
        self.assertGreaterEqual(valor_corregido, 0.5)
    
    def test_decay_statistics(self):
        """Test: Cálculo correcto de estadísticas de decay"""
        # Configurar parámetros con diferentes desviaciones
        self.learning_system.parametros_aprendidos.update({
            'sensibilidad_primeros_cuartos': 1.2,  # 20% desviación
            'sensibilidad_tercer_cuarto': 0.95,  # 5% desviación
            'conservadurismo_q4': 1.8,  # 80% desviación
        })
        
        # Calcular estadísticas
        stats = self.learning_system._calcular_estadisticas_decay()
        
        # Verificar cálculos
        self.assertEqual(stats['estado'], 'activo')
        self.assertEqual(stats['parametros_con_decay'], 3)
        self.assertAlmostEqual(stats['desviacion_promedio_pct'], 35.0, places=1)
        self.assertEqual(stats['salud_decay'], 'regular')  # >20% y <50%

if __name__ == '__main__':
    unittest.main()
```

### 2. Pruebas de Integración

#### 2.1 Flujo Completo de Predicción

**Archivo**: `test_integration_prediction_flow.py`

```python
import unittest
import asyncio
from unittest.mock import Mock, patch
from core.live_api import LiveProcessor

class TestIntegrationPredictionFlow(unittest.TestCase):
    
    async def test_complete_prediction_flow(self):
        """Test: Flujo completo de predicción con mejoras integradas"""
        
        # Configurar mocks
        with patch('core.live_api.SofaScoreClient') as mock_client:
            # Mock de SofaScoreClient
            mock_instance = Mock()
            mock_client.return_value = mock_instance
            
            # Mock de datos de juego
            mock_instance.get_live_game_data.return_value = {
                'q_scores': {
                    'q1_home': 25, 'q1_away': 23,
                    'q2_home': 28, 'q2_away': 26,
                    'q3_home': 30, 'q3_away': 28
                },
                'time_info': {
                    'time_into_quarter': 180,  # 3 minutos en Q3
                    'minutes_played': 27.0,
                    'current_quarter_display': 'Q3'
                },
                'context': {
                    'estimated_quarter': 'Q3',
                    'minutes_played': 27.0,
                    'current_pace': 105.5
                }
            }
            
            # Mock de estadísticas
            mock_instance.get_game_statistics.return_value = {
                'home_field_goals_made': 30,
                'away_field_goals_made': 28,
                'home_personal_fouls': 8,
                'away_personal_fouls': 10
            }
            
            # Configurar processor
            trained_data = {
                'league_name': 'NBA',
                'team_names': ['Lakers', 'Celtics']
            }
            
            processor = LiveProcessor(trained_data)
            await processor.initialize()
            
            # Ejecutar procesamiento completo
            game_info = {
                'id': 'test_game',
                'home_team': 'Lakers',
                'away_team': 'Celtics',
                'status': 'Q3 - 3:00'
            }
            
            result = await processor.process_game_update('test_game', game_info)
            
            # Verificaciones
            self.assertIsNotNone(result)
            
            # Verificar que se usó tiempo real en adaptive split detector
            live_signals = result.get('live_signals', {})
            adaptive_split = live_signals.get('adaptive_split_detector', 0)
            self.assertGreater(adaptive_split, 0)  # Debe detectar cambio con 30 pts en 3 min
            
            # Verificar que se usó ContextAnalyzer en run detector
            run_signals = live_signals.get('run_active', False)
            self.assertTrue(run_signals.get('context_used', False))
            
            # Verificar que la predicción se generó correctamente
            prediction = result.get('prediction', 0)
            self.assertGreater(prediction, 150)  # Predicción razonable para totals

if __name__ == '__main__':
    unittest.main()
```

### 3. Pruebas de Regresión

#### 3.1 Comparación de Precisión

**Archivo**: `test_regression_accuracy.py`

```python
import unittest
import pandas as pd
from core.live_features import calculate_adaptive_split_detector

class TestRegressionAccuracy(unittest.TestCase):
    
    def test_ppm_calculation_accuracy(self):
        """Test: Precisión del cálculo de PPM con tiempo real"""
        
        # Casos de prueba conocidos
        test_cases = [
            {
                'name': 'Q3 temprano (2 min)',
                'q_scores': {
                    'q1_home': 25, 'q1_away': 23,
                    'q2_home': 28, 'q2_away': 26,
                    'q3_home': 8, 'q3_away': 7
                },
                'time_info': {'time_into_quarter': 120},
                'expected_ppm': 7.5,  # 15 pts / 2 min
                'should_detect': True
            },
            {
                'name': 'Q3 medio (6 min)',
                'q_scores': {
                    'q1_home': 25, 'q1_away': 23,
                    'q2_home': 28, 'q2_away': 26,
                    'q3_home': 24, 'q3_away': 22
                },
                'time_info': {'time_into_quarter': 360},
                'expected_ppm': 7.67,  # 46 pts / 6 min
                'should_detect': True
            },
            {
                'name': 'Q3 completo (12 min)',
                'q_scores': {
                    'q1_home': 25, 'q1_away': 23,
                    'q2_home': 28, 'q2_away': 26,
                    'q3_home': 35, 'q3_away': 33
                },
                'time_info': {'time_into_quarter': 720},
                'expected_ppm': 5.67,  # 68 pts / 12 min
                'should_detect': False  # No es tan diferente de Q2
            }
        ]
        
        for case in test_cases:
            with self.subTest(case=case['name']):
                result = calculate_adaptive_split_detector(
                    case['q_scores'], 'q3_progress', time_info=case['time_info']
                )
                
                if case['should_detect']:
                    self.assertGreater(result, 0.5, 
                                   f"Debe detectar cambio en {case['name']}")
                else:
                    self.assertLess(result, 0.5, 
                                  f"No debe detectar cambio en {case['name']}")

if __name__ == '__main__':
    unittest.main()
```

### 4. Pruebas de Carga y Estrés

#### 4.1 Rendimiento con Múltiples Partidos

**Archivo**: `test_stress_multiple_games.py`

```python
import unittest
import asyncio
import time
from unittest.mock import Mock, patch
from core.live_api import LiveProcessor

class TestStressMultipleGames(unittest.TestCase):
    
    async def test_multiple_games_performance(self):
        """Test: Rendimiento con múltiples partidos simultáneos"""
        
        # Configurar mocks para múltiples partidos
        games_data = []
        for i in range(10):  # 10 partidos simultáneos
            games_data.append({
                'id': f'game_{i}',
                'home_team': f'Team_{i}_A',
                'away_team': f'Team_{i}_B',
                'status': 'Q3 - 3:00'
            })
        
        with patch('core.live_api.SofaScoreClient') as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance
            
            # Mock de respuestas
            mock_instance.get_live_game_data.return_value = {
                'q_scores': {
                    'q1_home': 25, 'q1_away': 23,
                    'q2_home': 28, 'q2_away': 26,
                    'q3_home': 30, 'q3_away': 28
                },
                'time_info': {
                    'time_into_quarter': 180,
                    'minutes_played': 27.0
                }
            }
            
            # Configurar processor
            trained_data = {'league_name': 'NBA', 'team_names': []}
            processor = LiveProcessor(trained_data)
            await processor.initialize()
            
            # Medir rendimiento
            start_time = time.time()
            
            # Procesar todos los juegos concurrentemente
            tasks = []
            for game in games_data:
                task = processor.process_game_update(game['id'], game)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Verificaciones
            self.assertLess(processing_time, 30.0,  # Máximo 30 segundos para 10 partidos
                                   f"Tiempo excedido: {processing_time:.2f}s")
            
            # Verificar que todos los resultados son exitosos
            successful_results = [r for r in results if not isinstance(r, Exception)]
            self.assertEqual(len(successful_results), len(games_data),
                           f"Errores en {len(results) - len(successful_results)} partidos")

if __name__ == '__main__':
    unittest.main()
```

## Ejecución de Pruebas

### 1. Configuración del Entorno

```bash
# Instalar dependencias de prueba
pip install pytest pytest-asyncio pytest-mock

# Configurar variables de entorno
export TEST_MODE=true
export TEST_CONFIG_PATH=test_config.json
```

### 2. Suite de Pruebas Completa

**Archivo**: `run_all_tests.py`

```python
import subprocess
import sys
import time

def run_test_suite(test_name, test_file):
    """Ejecuta una suite de pruebas y retorna resultados"""
    print(f"\n{'='*60}")
    print(f"EJECUTANDO: {test_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            test_file, '-v', '--tb=short'
        ], capture_output=True, text=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        success = result.returncode == 0
        
        return {
            'name': test_name,
            'success': success,
            'duration': duration,
            'output': result.stdout,
            'errors': result.stderr
        }
    except Exception as e:
        return {
            'name': test_name,
            'success': False,
            'duration': 0,
            'output': '',
            'errors': str(e)
        }

def main():
    """Ejecuta todas las suites de pruebas"""
    test_suites = [
        ('Adaptive Split Detector', 'test_adaptive_split_detector.py'),
        ('Run Detector con ContextAnalyzer', 'test_run_detector_context.py'),
        ('Sistema de Aprendizaje con Decay', 'test_learning_system_decay.py'),
        ('Integración de Flujo Completo', 'test_integration_prediction_flow.py'),
        ('Regresión de Precisión', 'test_regression_accuracy.py'),
        ('Estrés con Múltiples Partidos', 'test_stress_multiple_games.py')
    ]
    
    results = []
    total_start_time = time.time()
    
    for test_name, test_file in test_suites:
        result = run_test_suite(test_name, test_file)
        results.append(result)
        
        # Mostrar resumen inmediato
        status = "✅ EXITO" if result['success'] else "❌ FALLO"
        print(f"\n{status} - {test_name}: {result['duration']:.2f}s")
        
        if not result['success']:
            print(f"Errores: {result['errors'][:200]}...")
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    # Resumen final
    print(f"\n{'='*60}")
    print("RESUMEN EJECUCIÓN DE PRUEBAS")
    print(f"{'='*60}")
    
    successful_tests = sum(1 for r in results if r['success'])
    total_tests = len(results)
    
    print(f"Pruebas exitosas: {successful_tests}/{total_tests}")
    print(f"Tiempo total: {total_duration:.2f}s")
    
    if successful_tests == total_tests:
        print("🎉 TODAS LAS PRUEBAS PASARON")
        return 0
    else:
        print("⚠️ HAY PRUEBAS FALLIDAS")
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
```

## Métricas de Validación

### 1. Métricas de Precisión

- **Adaptive Split Detector**: Precisión en detección de cambios de pace
- **Run Detector**: Precisión en detección de rachas con contexto
- **Sistema de Aprendizaje**: Convergencia hacia valores óptimos

### 2. Métricas de Rendimiento

- **Tiempo de respuesta**: Latencia en procesamiento de predicciones
- **Throughput**: Predicciones por segundo bajo carga
- **Uso de memoria**: Consumo de recursos del sistema

### 3. Métricas de Estabilidad

- **Consistencia**: Variabilidad en predicciones similares
- **Recuperación**: Capacidad de recuperación ante errores
- **Robustez**: Comportamiento bajo condiciones extremas

## Criterios de Aceptación

1. **Bug corregido**: Adaptive Split Detector usa tiempo real correctamente
2. **Mejoras funcionales**: Run Detector se integra con ContextAnalyzer
3. **Decay activo**: Sistema de aprendizaje aplica decay correctamente
4. **Sin regresión**: Precisión general no disminuye
5. **Rendimiento aceptable**: Tiempo de respuesta < 5 segundos por predicción
6. **Estabilidad confirmada**: Sistema estable bajo carga

## Plan de Implementación

1. **Semana 1**: Implementar pruebas unitarias y ejecutar validación inicial
2. **Semana 2**: Ejecutar pruebas de integración y corregir problemas
3. **Semana 3**: Realizar pruebas de regresión y ajustes finos
4. **Semana 4**: Ejecutar pruebas de estrés y optimización de rendimiento

## Documentación de Resultados

Todos los resultados de pruebas se documentarán en:

- `test_results/unit_tests.json`
- `test_results/integration_tests.json`
- `test_results/performance_metrics.json`
- `test_results/validation_summary.md`

## Conclusión

Este plan de pruebas y validación asegura que las mejoras implementadas funcionen correctamente, no introduzcan regresiones y mejoren la calidad general del sistema de predicción de BasketMLG.