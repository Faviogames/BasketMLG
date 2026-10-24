# ===========================================
# Archivo: core/training.py (v4.0 - SISTEMA UNIFICADO)
# Entrena RF y XGBoost simultáneamente con selección de modelo
# ===========================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import joblib
import json
import os
import numpy as np

# Imports del proyecto
from config import DATA_FOLDER, MODELS_FOLDER, FEATURES_TO_USE, UNIVERSALLY_USELESS_FEATURES, FILTERED_FEATURES_TO_USE, HISTORICAL_FEATURES_TO_USE
from core.features import calculate_features
from core.data_processing import load_league_data

# Verificar disponibilidad de XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost no instalado. Ejecuta: pip install xgboost")

# Sistema dual integrado directamente (no necesita import externo)
DUAL_SYSTEM_AVAILABLE = XGBOOST_AVAILABLE  # Solo depende de XGBoost

# ===========================================
# SISTEMA DUAL INTEGRADO
# ===========================================

class DualModelSystem:
    """Sistema que maneja RF y XGBoost simultáneamente"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def get_available_models(self):
        """Retorna lista de modelos disponibles"""
        available = []
        if XGBOOST_AVAILABLE:
            available.append("xgboost")
        return available
    
    def create_model(self, model_type, league_name, custom_params=None):
        """Crea el modelo especificado con parámetros optimizados"""

        if model_type == "xgboost":
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost no está instalado")
            
            # Parámetros por defecto para basketball over/under
            default_params = {
                'n_estimators': 300,
                'learning_rate': 0.1,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'n_jobs': -1,
                'eval_metric': 'rmse'
            }

            # Optuna parameter loading removed by request. Using default XGBoost parameters.

            if custom_params:
                default_params.update(custom_params)
            return xgb.XGBRegressor(**default_params)
        
        else:
            raise ValueError(f"Modelo '{model_type}' no soportado")
    
    def train_all_models(self, X_train, y_train, X_test, y_test, league_name):
        """Entrena todos los modelos disponibles y evalúa"""
        available_models = self.get_available_models()
        
        print(f"\n[INFO] Entrenando {len(available_models)} modelos...")
        
        for model_type in available_models:
            print(f"\n[INFO] Entrenando {model_type.upper()}...")
            
            # Crear y entrenar modelo
            model = self.create_model(model_type, league_name)
            model.fit(X_train, y_train)
            
            # Evaluar
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            residuals = y_test - y_pred
            std_dev = np.std(residuals)
            
            # Guardar resultados
            self.models[model_type] = model
            self.results[model_type] = {
                'model': model,
                'mae': mae,
                'std_dev': std_dev,
                'model_type': model_type
            }
            
            print(f"   [OK] {model_type.upper()}: MAE = {mae:.3f}, Std Dev = {std_dev:.3f}")
        
        # Mostrar comparación
        self._show_comparison()
        
        return self.results
    
    def _show_comparison(self):
        """Muestra comparación entre modelos"""
        if len(self.results) <= 1:
            return
        
        print(f"\n📊 COMPARACIÓN DE MODELOS:")
        print("-" * 50)
        
        # Ordenar por MAE
        sorted_results = sorted(self.results.items(), key=lambda x: x[1]['mae'])
        
        for i, (model_type, result) in enumerate(sorted_results):
            rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
            print(f"{rank} {model_type.upper()}: MAE {result['mae']:.3f}")
        
        # Mostrar mejora
        if len(sorted_results) >= 2:
            best_mae = sorted_results[0][1]['mae']
            second_mae = sorted_results[1][1]['mae']
            improvement = ((second_mae - best_mae) / second_mae) * 100
            
            print(f"\n💡 MEJOR MODELO: {sorted_results[0][0].upper()}")
            print(f"📈 Mejora sobre el segundo: {improvement:.1f}%")
    
    def get_best_model(self):
        """Retorna el mejor modelo basado en MAE"""
        if not self.results:
            return None, None
        
        best_model_type = min(self.results.keys(), key=lambda k: self.results[k]['mae'])
        return best_model_type, self.results[best_model_type]
    
    def select_model_interactively(self):
        """Permite al usuario seleccionar el modelo manualmente (solo XGBoost disponible)"""
        if len(self.results) <= 1:
            return list(self.results.keys())[0] if self.results else None

        print(f"\n🤖 SELECCIÓN DE MODELO:")
        print("=" * 50)
        print("Nota: Solo XGBoost está disponible (Random Forest eliminado)")

        # Mostrar XGBoost result
        result = self.results["xgboost"]
        print(f"1. 🥇 XGBOOST")
        print(f"   📊 MAE: {result['mae']:.3f}")
        print(f"   📈 Std Dev: {result['std_dev']:.3f}")
        print()

        # Solo opción automática
        print(f"A. 🤖 AUTOMÁTICO (usar XGBoost)")
        print()

        # Solicitar selección
        while True:
            try:
                choice = input("🎯 Elige tu modelo (1, A para automático): ")

                if choice.upper() == 'A' or choice == '1':
                    print(f"✅ Seleccionado: XGBOOST")
                    return "xgboost"
                else:
                    print(f"❌ Opción inválida. Elige 1 o A")

            except ValueError:
                print(f"❌ Entrada inválida. Elige 1 o A")
            except KeyboardInterrupt:
                print(f"\n⚠️ Selección cancelada. Usando XGBoost automáticamente.")
                return "xgboost"
    
    def get_model_by_preference(self, preference="auto"):
        """
        Obtiene modelo según preferencia
        preference: "auto" | "xgboost" | "interactive"
        """
        if preference == "auto":
            return self.get_best_model()
        elif preference == "interactive":
            selected_type = self.select_model_interactively()
            return selected_type, self.results[selected_type]
        elif preference in self.results:
            return preference, self.results[preference]
        else:
            print(f"⚠️ Modelo '{preference}' no disponible. Usando XGBoost.")
            return "xgboost", self.results["xgboost"]
    
    def save_models(self, league_name, models_folder="./models", preferred_model=None):
        """Guarda todos los modelos entrenados y permite selección manual"""
        if not os.path.exists(models_folder):
            os.makedirs(models_folder)
        
        safe_league_name = league_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        saved_files = []
        
        # Determinar modelo a usar como principal
        if preferred_model and preferred_model in self.results:
            primary_model = preferred_model
            print(f"🎯 Usando modelo seleccionado manualmente: {primary_model.upper()}")
        else:
            primary_model = self.get_best_model()[0]
            print(f"🏆 Usando mejor modelo automático: {primary_model.upper()}")
        
        for model_type, result in self.results.items():
            # Nombre del archivo
            filename = f"{safe_league_name}_{model_type}.joblib"
            filepath = os.path.join(models_folder, filename)
            
            # Datos a guardar
            model_data = {
                'model': result['model'],
                'model_type': model_type,
                'mae': result['mae'],
                'std_dev': result['std_dev'],
                'league_name': league_name,
                'is_primary': model_type == primary_model,
                'features_used': getattr(self, 'features_used', []),
                'historical_df': getattr(self, 'historical_df', None),
                'team_names': getattr(self, 'team_names', []),
                'impute_values': getattr(self, 'impute_values', {}),
                'audit_results': getattr(self, 'audit_results', None)  # ✅ AÑADIDO: Resultados de auditoría
            }
            
            # Guardar
            joblib.dump(model_data, filepath)
            saved_files.append(filepath)
            
            # Indicar cuál es el principal
            primary_indicator = "⭐ PRINCIPAL" if model_type == primary_model else ""
            print(f"💾 {model_type.upper()} guardado: {filename} {primary_indicator}")
        
        return saved_files, primary_model

# ===========================================
# FUNCIONES PRINCIPALES DE ENTRENAMIENTO
# ===========================================



def analyze_feature_importance(model, feature_names, league_name):
    """
    🔍 Analiza qué features realmente usa el modelo
    """
    if not hasattr(model, 'feature_importances_'):
        print(f"⚠️ El modelo {type(model).__name__} no tiene 'feature_importances_'. Saltando auditoría.")
        return {
            'feature_importance': [],
            'useless_features': [],
            'stats': {
                'total_features': len(feature_names),
                'top_10_concentration': 0.0,
            }
        }

    print(f"\n{'='*60}")
    print(f"🔍 AUDITORÍA DE FEATURES - {league_name}")
    print(f"{'='*60}")
    
    importances = model.feature_importances_
    feature_importance = list(zip(feature_names, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    # TOP 15 features más importantes
    print(f"\n🎯 TOP 15 FEATURES MÁS IMPORTANTES:")
    total_importance_top15 = 0
    for i, (feature, importance) in enumerate(feature_importance[:15]):
        total_importance_top15 += importance
        print(f"{i+1:2d}. {feature:<40} {importance:.4f} ({importance*100:.1f}%)")
    
    print(f"\n📊 Las top 15 features representan {total_importance_top15*100:.1f}% del poder predictivo")
    
    # Features inútiles
    useless_threshold = 0.001
    useless_features = [f for f, imp in feature_importance if imp < useless_threshold]
    
    if useless_features:
        print(f"\n🗑️ FEATURES POTENCIALMENTE INÚTILES ({len(useless_features)} features con <0.1% importancia):")
        for i, feature in enumerate(useless_features[:10]):  # Solo primeros 10
            importance = next(imp for f, imp in feature_importance if f == feature)
            print(f"{i+1:2d}. {feature:<40} {importance:.6f}")
        
        if len(useless_features) > 10:
            print(f"... y {len(useless_features) - 10} más")
    else:
        print(f"\n✅ Todas las features tienen importancia significativa (>0.1%)")
    
    return {
        'feature_importance': feature_importance,
        'useless_features': useless_features,
        'stats': {
            'total_features': len(feature_names),
            'useful_features': len(feature_names) - len(useless_features),
            'useless_features': len(useless_features),
            'top_10_concentration': sum(imp for _, imp in feature_importance[:10]),
            'top_25_concentration': sum(imp for _, imp in feature_importance[:25])
        }
    }

def _ensure_match_group_key(df: pd.DataFrame) -> pd.Series:
    """
    Deriva una clave de agrupación por partido para evitar fugas entre halftime y q3_end del mismo juego.
    Prioridad: raw_match.match_id -> fallback (home_team|away_team|raw_date) -> pares por índice.
    """
    if 'raw_match' in df.columns:
        def _key_row(row):
            try:
                rm = row['raw_match']
                mid = rm.get('match_id') if isinstance(rm, dict) else None
            except Exception:
                mid = None
            if mid:
                return str(mid)
            # Fallback compuesto estable
            home = str(row.get('home_team', ''))
            away = str(row.get('away_team', ''))
            try:
                date_val = ''
                if isinstance(rm, dict) and 'date' in rm:
                    date_val = str(rm['date'])
            except Exception:
                date_val = ''
            return f"{home}|{away}|{date_val}"
        groups = df.apply(_key_row, axis=1)
    else:
        # Último recurso: emparejar filas (asumiendo 2 filas por partido)
        groups = pd.Series((np.arange(len(df)) // 2).astype(str), index=df.index)
    return groups

def _create_base_model(model_type: str):
    """
    Crea un modelo base consistente con la inferencia live.
    Solo XGBoost - Random Forest eliminado.
    """
    if model_type == "xgboost" and XGBOOST_AVAILABLE:
        # Parámetros por defecto utilizados en create_model
        return xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            eval_metric='rmse'
        )
    else:
        raise ValueError("Solo XGBoost está soportado. Random Forest ha sido eliminado.")

def _generate_oof_base_predictions(
    df: pd.DataFrame,
    feature_cols: list,
    label_col: str,
    base_model_type: str,
    league_name: str,
    n_splits: int = 5
) -> pd.DataFrame:
    """
    Genera predicciones OOF base agrupadas por partido para evitar fuga de datos.
    - Solo usa XGBoost (Random Forest eliminado)
    - Divide cronológicamente por orden de aparición de los partidos (grupos contiguos).
    - Entrena con grupos anteriores y valida con el siguiente bloque.
    """
    if base_model_type != "xgboost":
        raise ValueError("Solo XGBoost está soportado para predicciones OOF base")

    df = df.copy()
    groups = _ensure_match_group_key(df)

    # Orden cronológico por primera aparición del grupo
    unique_groups_ordered = []
    seen = set()
    for g in groups:
        if g not in seen:
            seen.add(g)
            unique_groups_ordered.append(g)
    n_groups = len(unique_groups_ordered)
    if n_groups < 2:
        print(f"⚠️ OOF saltado para {league_name}: grupos insuficientes ({n_groups})")
        df['base_pred'] = df[feature_cols].mean(axis=1).fillna(0.0)  # fallback trivial
        return df

    n_splits_eff = min(max(2, n_groups), n_splits)
    split_idx = np.linspace(0, n_groups, n_splits_eff + 1, dtype=int)

    oof_pred = pd.Series(index=df.index, dtype=float)
    used_folds = 0

    for k in range(1, len(split_idx)):
        train_end = split_idx[k - 1]
        val_start = split_idx[k - 1]
        val_end = split_idx[k]

        if train_end == 0:
            # Evitar fold sin datos de entrenamiento
            continue

        train_groups = set(unique_groups_ordered[:train_end])
        val_groups = set(unique_groups_ordered[val_start:val_end])

        train_idx = df.index[groups.isin(train_groups)]
        val_idx = df.index[groups.isin(val_groups)]

        if len(train_idx) == 0 or len(val_idx) == 0:
            continue

        model = _create_base_model(base_model_type)

        X_train = df.loc[train_idx, feature_cols].copy()
        y_train = df.loc[train_idx, label_col].copy()
        X_val = df.loc[val_idx, feature_cols].copy()

        # Imputación por medias del fold de entrenamiento
        impute = X_train.mean(numeric_only=True).to_dict()
        X_train = X_train.fillna(value=impute)
        X_val = X_val.fillna(value=impute)

        model.fit(X_train, y_train)
        oof_pred.loc[val_idx] = model.predict(X_val)
        used_folds += 1

    # Completar posibles huecos con modelo entrenado en todo el dataset
    if oof_pred.isnull().any():
        model_all = _create_base_model(base_model_type)
        impute_all = df[feature_cols].mean(numeric_only=True).to_dict()
        X_all = df[feature_cols].fillna(value=impute_all)
        y_all = df[label_col]
        model_all.fit(X_all, y_all)
        missing_idx = oof_pred[oof_pred.isnull()].index
        if len(missing_idx) > 0:
            oof_pred.loc[missing_idx] = model_all.predict(X_all.loc[missing_idx])

    df['base_pred'] = oof_pred.values
    try:
        base_mae = mean_absolute_error(df[label_col], df['base_pred'])
        print(f"🧪 OOF base (XGBoost) [{league_name}] MAE = {base_mae:.3f} | folds {used_folds}/{n_splits_eff}")
    except Exception:
        pass
    return df

def _simulate_historical_signals(row: pd.Series, df: pd.DataFrame) -> dict:
    """
    Simula señales live basadas en datos históricos para entrenamiento del meta-modelo.
    Esto permite que el meta-modelo aprenda patrones de ajuste de señales.
    """
    try:
        # Extraer datos del partido
        final_score = row.get('final_total_score', 0)
        halftime_score = row.get('halftime_total', 0)
        q3_score = row.get('q3_end_total', 0)

        # Simular señales basadas en patrones históricos
        signals = {}

        # 1. Simular A/TO ratio (basado en turnovers y assists disponibles)
        home_to = row.get('home_turnovers', 0)
        away_to = row.get('away_turnovers', 0)
        home_ast = row.get('home_assists', 0)
        away_ast = row.get('away_assists', 0)

        if home_to + away_to > 0:
            total_to = home_to + away_to
            total_ast = home_ast + away_ast
            ato_ratio = total_ast / total_to if total_to > 0 else 1.0
            # Normalizar vs league average (aprox 1.5-2.0)
            league_avg_ato = 1.8
            signals['diff_ato_ratio'] = (ato_ratio - league_avg_ato) / league_avg_ato
        else:
            signals['diff_ato_ratio'] = 0.0

        # 2. Simular FTI basado en FT attempts (proxy para foul trouble)
        home_fta = row.get('home_free_throws_attempted', 0)
        away_fta = row.get('away_free_throws_attempted', 0)
        total_fta = home_fta + away_fta

        # Estimar minutos jugados basados en score
        estimated_minutes = min(48, max(24, halftime_score / 1.8))  # Score halftime / rate
        ft_rate = total_fta / estimated_minutes if estimated_minutes > 0 else 0

        # FTI basado en FT rate (más FT = más foul trouble)
        signals['home_fti'] = min(1.0, max(0.0, (ft_rate * 0.6 - 0.5) / 1.5))
        signals['away_fti'] = signals['home_fti']  # Simplificación

        # 3. Simular TS% basado en eficiencia de tiro
        home_pts = row.get('home_score', halftime_score / 2)
        away_pts = row.get('away_score', halftime_score / 2)
        total_pts = home_pts + away_pts

        # Estimar TS% basado en puntos y posesiones aproximadas
        estimated_possessions = total_pts / 2.1  # Aprox possessions
        if estimated_possessions > 0:
            ts_estimate = total_pts / (2 * (estimated_possessions * 0.9))  # Aprox TS formula
            signals['diff_ts_live'] = (ts_estimate - 0.55) / 0.55  # vs league avg 55%
        else:
            signals['diff_ts_live'] = 0.0

        # 4. Simular run detector (basado en momentum entre cuartos)
        if halftime_score > 0 and q3_score > halftime_score:
            run_strength = min(1.0, (q3_score - halftime_score) / halftime_score)
            signals['run_active'] = True
            signals['run_strength'] = run_strength
            signals['run_side'] = 'home' if home_pts > away_pts else 'away'
        else:
            signals['run_active'] = False
            signals['run_strength'] = 0.0
            signals['run_side'] = 'none'

        # 5. Simular rebound differential
        home_orb = row.get('home_offensive_rebounds', 0)
        away_orb = row.get('away_offensive_rebounds', 0)
        home_drb = row.get('home_defensive_rebounds', 0)
        away_drb = row.get('away_defensive_rebounds', 0)

        total_reb = home_orb + away_orb + home_drb + away_drb
        if total_reb > 0:
            home_reb_pct = (home_orb + home_drb) / total_reb
            signals['home_treb_diff'] = (home_reb_pct - 0.5) / 0.5  # vs 50%
            signals['away_treb_diff'] = -signals['home_treb_diff']
        else:
            signals['home_treb_diff'] = 0.0
            signals['away_treb_diff'] = 0.0

        # Calcular métricas agregadas
        from analysis.live_mode import calculate_signal_strength, count_active_signals, calculate_signal_adjustment_magnitude

        signal_strength = calculate_signal_strength(signals)
        active_signals = count_active_signals(signals)

        # Simular predicción con señales aplicadas (aproximación)
        base_pred = row.get('base_pred', final_score)
        # Simular ajuste basado en señales activas
        adjustment_factor = 1.0 + (signal_strength * 0.1) + (active_signals * 0.02)
        context_adjusted_pred = base_pred * adjustment_factor

        return {
            'signal_strength': signal_strength,
            'active_signals': active_signals,
            'adjustment_magnitude': abs(adjustment_factor - 1.0),
            'context_adjusted_pred': context_adjusted_pred
        }

    except Exception as e:
        # Fallback: señales neutras
        return {
            'signal_strength': 0.0,
            'active_signals': 0,
            'adjustment_magnitude': 0.0,
            'context_adjusted_pred': row.get('base_pred', row.get('final_total_score', 200))
        }

# META-MODELO ELIMINADO - Función completamente eliminada
# El meta-modelo será reemplazado por un sistema unificado inteligente

def train_single_model_legacy(X_train, y_train, X_test, y_test):
    """
    Entrenamiento legacy con XGBoost (fallback)
    """
    if XGBOOST_AVAILABLE:
        print("🚀 Entrenando XGBoost (modo legacy)...")

        # XGBoost parameters
        model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            eval_metric='rmse'
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        residuals = y_test - y_pred
        std_dev = np.std(residuals)

        print(f"✅ XGBoost: MAE = {mae:.3f}, Std Dev = {std_dev:.3f}")

        return {
            'model': model,
            'mae': mae,
            'std_dev': std_dev,
            'model_type': 'xgboost'
        }
    else:
        print("❌ XGBoost no disponible - no se puede entrenar modelo")
        raise ImportError("XGBoost no está instalado. Instala con: pip install xgboost")

def calculate_league_scoring_stats(df: pd.DataFrame) -> dict:
    """Calcula estadísticas de puntuación de la liga para overs/unders analysis."""
    if 'final_total_score' not in df.columns:
        return {}

    total_scores = df['final_total_score'].dropna()

    if total_scores.empty:
        return {}

    # Estadísticas básicas - convertir a float nativo de Python
    avg_points = float(total_scores.mean())
    median_points = float(total_scores.median())
    min_points = float(total_scores.min())
    max_points = float(total_scores.max())

    # Quartiles - convertir a float nativo de Python
    q1 = float(total_scores.quantile(0.25))
    q3 = float(total_scores.quantile(0.75))

    # Overs/unders analysis para líneas comunes
    common_lines = [200, 210, 220, 230, 240]
    overs_analysis = {}

    for line in common_lines:
        overs_count = int((total_scores > line).sum())  # Convertir a int nativo
        total_games = len(total_scores)
        overs_percentage = float(overs_count / total_games) if total_games > 0 else 0.0
        overs_analysis[f"over_{line}"] = round(overs_percentage, 3)

    # Determinar bias de la liga
    over_220_pct = overs_analysis.get("over_220", 0.5)
    league_bias = "overs_favored" if over_220_pct > 0.55 else "unders_favored" if over_220_pct < 0.45 else "neutral"

    return {
        "average_total_points": round(avg_points, 2),
        "median_total_points": round(median_points, 2),
        "scoring_distribution": {
            "min": round(min_points, 1),
            "max": round(max_points, 1),
            "q1": round(q1, 1),
            "q3": round(q3, 1)
        },
        "overs_unders_tendency": overs_analysis,
        "league_bias": league_bias
    }

def calculate_feature_analysis(audit_results: dict) -> dict:
    """Extrae análisis de features del audit_results."""
    if not audit_results or 'feature_importance' not in audit_results:
        return {}

    feature_importance = audit_results['feature_importance']

    # Top 10 features con categorías
    top_features = []
    for i, (feature_name, importance) in enumerate(feature_importance[:10]):
        # Categorizar features
        if any(keyword in feature_name.lower() for keyword in ['total', 'score', 'points', 'halftime', 'q1', 'q2', 'q3', 'q4']):
            category = "scoring"
        elif any(keyword in feature_name.lower() for keyword in ['pace', 'possessions', 'efficiency']):
            category = "pace"
        elif any(keyword in feature_name.lower() for keyword in ['turnover', 'assist', 'rebound']):
            category = "efficiency"
        else:
            category = "defensive"

        top_features.append({
            "name": feature_name,
            "importance": round(float(importance), 4),
            "category": category

        })

    # Categorías de features
    category_counts = {}
    for feature in top_features:
        cat = feature["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

    # Convertir a porcentajes
    total_top = len(top_features)
    feature_categories = {cat: round(count/total_top, 2) for cat, count in category_counts.items()}

    return {
        "top_10_features": top_features,
        "useless_features": audit_results.get('useless_features', []),
        "feature_categories": feature_categories
    }

def calculate_training_insights(model, X_test, y_test, audit_results: dict) -> dict:
    """Calcula métricas adicionales de rendimiento del modelo."""
    if not hasattr(model, 'predict'):
        return {}

    try:
        y_pred = model.predict(X_test)

        # R² score - convertir a float nativo
        from sklearn.metrics import r2_score
        r2 = float(r2_score(y_test, y_pred))

        # Explained variance - convertir a float nativo
        from sklearn.metrics import explained_variance_score
        explained_var = float(explained_variance_score(y_test, y_pred))

        # Median absolute error - convertir a float nativo
        median_ae = float(np.median(np.abs(y_test - y_pred)))

        # Distribución de predicciones - convertir a float nativo
        residuals = y_test - y_pred
        under_predictions = float((residuals > 0).mean())  # Modelo predijo menos que realidad
        accurate_predictions = float((np.abs(residuals) <= 5).mean())  # Dentro de 5 puntos
        over_predictions = float((residuals < 0).mean())  # Modelo predijo más que realidad

        # Estabilidad del modelo - convertir a float nativo
        if audit_results and 'stats' in audit_results:
            stability_score = float(audit_results['stats'].get('top_10_concentration', 0.5))
        else:
            stability_score = 0.5

        return {
            "performance_metrics": {
                "r2_score": round(r2, 3),
                "explained_variance": round(explained_var, 3),
                "median_absolute_error": round(median_ae, 2)
            },
            "prediction_distribution": {
                "under_predictions": round(under_predictions, 3),
                "accurate_predictions": round(accurate_predictions, 3),
                "over_predictions": round(over_predictions, 3)
            },
            "model_stability": {
                "feature_stability_score": round(stability_score, 3)
            }
        }
    except Exception as e:
        print(f"⚠️ Error calculando training insights: {e}")
        return {}

def calculate_scoring_patterns(df: pd.DataFrame) -> dict:
    """Calcula patrones de puntuación específicos de la liga."""
    if df.empty:
        return {}

    patterns = {}

    # Análisis de pace
    if 'pace_game' in df.columns:
        avg_pace = df['pace_game'].mean()
        if not pd.isna(avg_pace):
            patterns["pace_analysis"] = {
                "average_possessions": round(float(avg_pace), 1),
                "pace_consistency": 0.8  # Placeholder - se puede calcular varianza
            }

    # Eficiencia promedio
    if 'final_total_score' in df.columns and 'pace_game' in df.columns:
        total_scores = df['final_total_score'].dropna()
        pace_values = df['pace_game'].dropna()

        if not total_scores.empty and not pace_values.empty:
            # Eficiencia aproximada = puntos por posesión
            pace_mean = float(pace_values.mean())
            avg_efficiency = float(total_scores.mean()) / pace_mean if pace_mean > 0 else 0.0
            if not pd.isna(avg_efficiency):
                patterns["efficiency_metrics"] = {
                    "league_avg_efficiency": round(avg_efficiency, 3),
                    "high_efficiency_threshold": round(avg_efficiency * 1.15, 3)
                }

    # Distribución por cuartos
    quarter_cols = ['q1_total', 'q2_total', 'q3_total', 'q4_total']
    if all(col in df.columns for col in quarter_cols):
        quarter_totals = df[quarter_cols].mean()
        total_avg = float(quarter_totals.sum())

        if total_avg > 0:
            quarter_distribution = {
                f"q{i+1}_percentage": round(float(quarter_totals.iloc[i]) / total_avg, 3)
                for i in range(4)
            }
            patterns["quarter_distribution"] = quarter_distribution

    return patterns

def train_models_by_league():
    """Carga los datos, entrena modelos por liga y los guarda."""
    print("[INFO] Iniciando entrenamiento...")
    
    if not os.path.exists(MODELS_FOLDER):
        os.makedirs(MODELS_FOLDER)

    all_data = load_league_data(DATA_FOLDER)
    
    # 🔎 Filtro opcional por ligas vía variable de entorno (e.g., LEAGUES_INCLUDE=NBA)
    leagues_include = os.getenv('LEAGUES_INCLUDE')
    if leagues_include:
        include_list = [s.strip() for s in leagues_include.split(',') if s.strip()]
        filtered = {k: v for k, v in all_data.items() if k in include_list}
        excluded = sorted([k for k in all_data.keys() if k not in filtered.keys()])
        print(f"🔎 LEAGUES_INCLUDE activo: {include_list}")
        if excluded:
            print(f"   Excluyendo ligas: {excluded}")
        all_data = filtered
    
    if not all_data:
        print("❌ No se encontraron datos para procesar")
        return


    global_audit_results = {}

    for league, league_data in all_data.items():
        try:
            df = calculate_features(league_data)
        except Exception as e:
            print(f"[ERROR] Error calculando features para {league}: {e}")
            continue
        
        num_valid_data_points = len(df)
        print(f"📊 Data points válidos para entrenamiento: {num_valid_data_points}")
        
        if df.empty or num_valid_data_points < 20:
            print(f"❌ No hay suficientes datos válidos para '{league}'. Se necesitan al menos 20. Saltando.")
            continue

        # Preparar datos
        # ✅ CRITICAL FIX: Use HISTORICAL_FEATURES_TO_USE to prevent data leakage (no live features)
        features_for_this_model = [f for f in HISTORICAL_FEATURES_TO_USE if f in df.columns]
        print(f"🗑️ Features eliminadas (Leakage/Useless): {len(FEATURES_TO_USE) - len(features_for_this_model)}")
        print(f"✅ Features usadas: {len(features_for_this_model)}")
        X = df[features_for_this_model]
        y = df['final_total_score']
        
        print(f"🎯 Features incluidas en el modelo: {len(features_for_this_model)}")

        # --- OOF base + Meta-Modelo (Supervisor) ---
        try:
            # Solo XGBoost para predicciones base (Random Forest eliminado)
            df = _generate_oof_base_predictions(
                df, features_for_this_model, 'final_total_score',
                "xgboost", league_name=league, n_splits=5
            )
            # META-MODELO ELIMINADO - Ya no se entrena meta-modelo
            meta_info = None
        except Exception as e:
            print(f"⚠️ Meta-modelo no entrenado para {league}: {e}")
            meta_info = None
        
        # División train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # --- ENTRENAMIENTO ---
        if DUAL_SYSTEM_AVAILABLE:
            print(f"\n[INFO] INICIANDO SISTEMA DUAL...")
            
            try:
                # Crear sistema dual
                dual_system = DualModelSystem()
                
                # Verificar modelos disponibles
                available_models = dual_system.get_available_models()
                print(f"📊 Modelos disponibles: {available_models}")
                if len(available_models) < 2:
                    print(f"⚠️ Solo {len(available_models)} modelo(s) disponible(s). Usando XGBoost directamente.")
                    # Create XGBoost model directly
                    dual_system = DualModelSystem()
                    dual_system.features_used = features_for_this_model
                    dual_system.historical_df = df
                    dual_system.team_names = list(set(df['home_team'].unique().tolist() + df['away_team'].unique().tolist()))
                    dual_system.impute_values = df[features_for_this_model].mean().to_dict()

                    # Train XGBoost directly
                    model = dual_system.create_model("xgboost", league)
                    model.fit(X_train, y_train)

                    # Evaluate
                    y_pred = model.predict(X_test)
                    mae = mean_absolute_error(y_test, y_pred)
                    residuals = y_test - y_pred
                    std_dev = np.std(residuals)

                    model_result = {
                        'model': model,
                        'mae': mae,
                        'std_dev': std_dev,
                        'model_type': 'xgboost'
                    }
                    selected_model_type = 'xgboost'
                else:
                    # Pasar datos adicionales al sistema
                    dual_system.features_used = features_for_this_model
                    dual_system.historical_df = df
                    dual_system.team_names = list(set(df['home_team'].unique().tolist() + df['away_team'].unique().tolist()))
                    dual_system.impute_values = df[features_for_this_model].mean().to_dict()
                    
                    # Entrenar todos los modelos
                    print(f"🚀 Entrenando {len(available_models)} modelos...")
                    all_results = dual_system.train_all_models(X_train, y_train, X_test, y_test, league)
                    
                    # --- SELECCIÓN DE MODELO ---
                    # 🎯 CONFIGURA AQUÍ TU PREFERENCIA
                    model_preference = "xgboost"  # Opciones: "auto", "random_forest", "xgboost", "interactive"
                    
                    print(f"\n🎯 Método de selección: {model_preference}")
                    
                    # Obtener modelo según preferencia
                    selected_model_type, model_result = dual_system.get_model_by_preference(model_preference)
                    
                    if not model_result:
                        print(f"❌ No se pudo seleccionar un modelo para {league}. Usando Random Forest.")
                        model_result = train_single_model_legacy(X_train, y_train, X_test, y_test)
                        selected_model_type = 'random_forest'
                    else:
                        print(f"\n🎯 MODELO SELECCIONADO PARA {league}: {selected_model_type.upper()}")
                
            except Exception as e:
                print(f"❌ Error en sistema dual: {e}")
                print("🔄 Fallback a entrenamiento legacy...")
                model_result = train_single_model_legacy(X_train, y_train, X_test, y_test)
                selected_model_type = 'random_forest'
        else:
            # Entrenamiento legacy
            model_result = train_single_model_legacy(X_train, y_train, X_test, y_test)
            selected_model_type = 'random_forest'
        
        # Extraer datos del resultado
        model = model_result['model']
        mae = model_result['mae']
        std_dev = model_result['std_dev']
        
        print(f"✅ Entrenamiento completado:")
        print(f"   Modelo: {selected_model_type.upper()}")
        print(f"   MAE: {mae:.2f}")
        print(f"   Std Dev: {std_dev:.2f}")
        
        # 🔍 AUDITORÍA DE FEATURES
        audit_results = analyze_feature_importance(model, features_for_this_model, f"{league} ({selected_model_type})")
        global_audit_results[f"{league}_{selected_model_type}"] = audit_results

        # 💾 PERSISTIR FEATURES INÚTILES EN JSON
        if audit_results.get('useless_features'):
            try:
                from config import save_useless_features_to_json
                save_useless_features_to_json(audit_results['useless_features'])
                print(f"💾 Features inútiles guardadas en JSON: {len(audit_results['useless_features'])} features")
            except Exception as e:
                print(f"⚠️ Error guardando features inútiles: {e}")
        
        # --- GUARDADO DE MODELOS ---
        team_names = list(set(df['home_team'].unique().tolist() + df['away_team'].unique().tolist()))
        
        # Crear valores de imputación
        impute_values = {}
        if not df[features_for_this_model].empty:
            impute_values = df[features_for_this_model].mean().to_dict()
        else:
            impute_values = {f: 0.0 for f in features_for_this_model}

        # Datos del modelo principal (compatibilidad con código existente)
        trained_data = {
            'model': model,
            'model_type': selected_model_type,
            'features_used': features_for_this_model,
            'std_dev': std_dev,
            'historical_df': df,
            'team_names': team_names,
            'league_name': league,
            'impute_values': impute_values,
            'audit_results': audit_results,
            'mae': mae
        }
        
        # Guardar modelo principal
        safe_league_name = league.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        model_filename = os.path.join(MODELS_FOLDER, f"{safe_league_name}.joblib")
        joblib.dump(trained_data, model_filename)

        # Guardar bundle JSON único con info base + meta (si existe)
        try:
            # Calcular métricas adicionales
            league_stats = calculate_league_scoring_stats(df)
            feature_analysis = calculate_feature_analysis(audit_results)
            training_insights = calculate_training_insights(model, X_test, y_test, audit_results)
            scoring_patterns = calculate_scoring_patterns(df)

            bundle_json = {
                'artifact_type': 'bundle',
                'league_name': league,
                'created_at': str(pd.Timestamp.now()),
                'base': {
                    'model_type': selected_model_type,
                    'mae': float(mae) if mae is not None else None,
                    'std_dev': float(std_dev) if std_dev is not None else None,
                    'teams_count': int(len(team_names)),
                    'matches_count': int(len(df)),
                    'features_count': int(len(features_for_this_model)),
                    'features_used': None  # evitar JSON enorme; activar si se desea
                }
            }

            # Agregar secciones adicionales si tienen datos
            if league_stats:
                bundle_json['league_stats'] = league_stats
            if feature_analysis:
                bundle_json['feature_analysis'] = feature_analysis
            if training_insights:
                bundle_json['training_insights'] = training_insights
            if scoring_patterns:
                bundle_json['scoring_patterns'] = scoring_patterns

            # Adjuntar meta payload completo dentro del bundle
            if 'meta_info' in locals() and meta_info:
                bundle_json['meta'] = meta_info

            bundle_json_path = os.path.join(MODELS_FOLDER, f"{safe_league_name}.json")
            with open(bundle_json_path, 'w', encoding='utf-8') as f:
                json.dump(bundle_json, f, ensure_ascii=False, indent=2)

            print(f"📊 JSON bundle mejorado guardado con {len(bundle_json) - 3} secciones adicionales")

        except Exception as _json_main_err:
            print(f"⚠️ No se pudo escribir JSON bundle del modelo: {_json_main_err}")
        
        print(f"[INFO] Modelo principal guardado: {safe_league_name}.joblib")
        
        # Guardar modelos duales si están disponibles
        if DUAL_SYSTEM_AVAILABLE and 'dual_system' in locals() and hasattr(dual_system, 'results') and dual_system.results:
            try:
                # ✅ PASAR AUDIT_RESULTS AL SISTEMA DUAL
                dual_system.audit_results = audit_results
                saved_files, primary_model = dual_system.save_models(league, MODELS_FOLDER, selected_model_type)
                print(f"💾 Modelos duales guardados: {len(saved_files)} archivos")
            except Exception as e:
                print(f"⚠️ Error guardando modelos duales: {e}")
        
        print(f"📊 Resumen: {len(team_names)} equipos | {len(df)} partidos | {len(features_for_this_model)} features")

    # 📋 RESUMEN GLOBAL DE AUDITORÍA
    print(f"\n{'='*80}")
    print(f"📋 RESUMEN GLOBAL DE AUDITORÍA")
    print(f"{'='*80}")
    
    total_leagues = len(global_audit_results)
    print(f"🏆 Ligas procesadas: {total_leagues}")
    
    if global_audit_results:
        # Estadísticas promedio
        avg_feature_count = np.mean([results['stats']['total_features'] for results in global_audit_results.values() if results.get('stats')])
        avg_concentration = np.mean([results['stats']['top_10_concentration'] for results in global_audit_results.values() if results.get('stats')])
        total_useless_features = sum(len(results['useless_features']) for results in global_audit_results.values())
        
        print(f"📊 Promedio de features por liga: {avg_feature_count:.1f}")
        print(f"🎯 Concentración promedio top 10: {avg_concentration*100:.1f}%")
        print(f"🗑️ Total features potencialmente inútiles: {total_useless_features}")
        
        if total_useless_features > 0:
            print(f"\n💡 RECOMENDACIÓN GLOBAL:")
            print(f"   Considera crear una versión optimizada eliminando features inútiles")
            print(f"   Beneficios esperados: +25-30% velocidad, menos overfitting")
        else:
            print(f"\n✅ EXCELENTE: Todas las features parecen contribuir al modelo")
    
    print(f"\n[INFO] Entrenamiento completado!")

# Información de modelos
MODEL_INFO = {
    "xgboost": {
        "name": "XGBoost",
        "description": "Gradient Boosting optimizado",
        "pros": ["Alta precisión", "Maneja features débiles", "Regularización"],
        "cons": ["Más parámetros", "Menos interpretable"]
    }
}

def get_model_info(model_type):
    """Retorna información detallada del modelo"""
    return MODEL_INFO.get(model_type, {})

if __name__ == "__main__":
    train_models_by_league()