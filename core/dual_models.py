# ===========================================
# Archivo: core/dual_models.py
# Sistema dual RF + XGBoost - Versión Simple
# ===========================================
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np
import joblib
import os

# Intentar importar XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost no instalado. Ejecuta: pip install xgboost")

class DualModelSystem:
    """Sistema que maneja RF y XGBoost simultáneamente"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def get_available_models(self):
        """Retorna lista de modelos disponibles"""
        available = ["random_forest"]
        if XGBOOST_AVAILABLE:
            available.append("xgboost")
        return available
    
    def create_model(self, model_type, custom_params=None):
        """Crea el modelo especificado con parámetros optimizados"""
        
        if model_type == "random_forest":
            default_params = {
                'n_estimators': 100,
                'random_state': 42,
                'n_jobs': -1
            }
            if custom_params:
                default_params.update(custom_params)
            return RandomForestRegressor(**default_params)
        
        elif model_type == "xgboost":
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost no está instalado")
            
            # Parámetros optimizados para basketball over/under
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
            if custom_params:
                default_params.update(custom_params)
            return xgb.XGBRegressor(**default_params)
        
        else:
            raise ValueError(f"Modelo '{model_type}' no soportado")
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """Entrena todos los modelos disponibles y evalúa"""
        available_models = self.get_available_models()
        
        print(f"\n🚀 Entrenando {len(available_models)} modelos...")
        
        for model_type in available_models:
            print(f"\n📊 Entrenando {model_type.upper()}...")
            
            # Crear y entrenar modelo
            model = self.create_model(model_type)
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
            
            print(f"   ✅ {model_type.upper()}: MAE = {mae:.3f}, Std = {std_dev:.3f}")
        
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
        """Permite al usuario seleccionar el modelo manualmente"""
        if len(self.results) <= 1:
            return list(self.results.keys())[0] if self.results else None
        
        print(f"\n🤖 SELECCIÓN DE MODELO:")
        print("=" * 50)
        
        # Mostrar opciones ordenadas por rendimiento
        sorted_results = sorted(self.results.items(), key=lambda x: x[1]['mae'])
        
        for i, (model_type, result) in enumerate(sorted_results, 1):
            rank = "🥇" if i == 1 else "🥈" if i == 2 else f"{i}."
            print(f"{i}. {rank} {model_type.upper()}")
            print(f"   📊 MAE: {result['mae']:.3f}")
            print(f"   📈 Std Dev: {result['std_dev']:.3f}")
            print()
        
        # Opción automática
        print(f"A. 🤖 AUTOMÁTICO (usar el mejor: {sorted_results[0][0].upper()})")
        print()
        
        # Solicitar selección
        while True:
            try:
                choice = input("🎯 Elige tu modelo (1-{}, A para automático): ".format(len(sorted_results)))
                
                if choice.upper() == 'A':
                    selected = sorted_results[0][0]
                    print(f"✅ Seleccionado automáticamente: {selected.upper()}")
                    return selected
                
                choice_num = int(choice)
                if 1 <= choice_num <= len(sorted_results):
                    selected = sorted_results[choice_num - 1][0]
                    print(f"✅ Seleccionado manualmente: {selected.upper()}")
                    return selected
                else:
                    print(f"❌ Opción inválida. Elige entre 1-{len(sorted_results)} o A")
                    
            except ValueError:
                print(f"❌ Entrada inválida. Elige un número (1-{len(sorted_results)}) o A")
            except KeyboardInterrupt:
                print(f"\n⚠️ Selección cancelada. Usando mejor modelo automáticamente.")
                return sorted_results[0][0]
    
    def get_model_by_preference(self, preference="auto"):
        """
        Obtiene modelo según preferencia
        preference: "auto" | "random_forest" | "xgboost" | "interactive"
        """
        if preference == "auto":
            return self.get_best_model()
        elif preference == "interactive":
            selected_type = self.select_model_interactively()
            return selected_type, self.results[selected_type]
        elif preference in self.results:
            return preference, self.results[preference]
        else:
            print(f"⚠️ Modelo '{preference}' no disponible. Usando automático.")
            return self.get_best_model()
    
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
                'impute_values': getattr(self, 'impute_values', {})
            }
            
            # Guardar
            joblib.dump(model_data, filepath)
            saved_files.append(filepath)
            
            # Indicar cuál es el principal
            primary_indicator = "⭐ PRINCIPAL" if model_type == primary_model else ""
            print(f"💾 {model_type.upper()} guardado: {filename} {primary_indicator}")
        
        return saved_files, primary_model

# Información de modelos
MODEL_INFO = {
    "random_forest": {
        "name": "Random Forest",
        "description": "Ensemble de árboles de decisión",
        "pros": ["Robusto", "Rápido", "Interpretable"],
        "cons": ["Puede overfittear", "Memoria intensivo"]
    },
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