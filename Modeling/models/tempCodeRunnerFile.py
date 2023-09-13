    def get_feature_importances(self):
        return self.xgb_model.feature_importances_

    def get_params(self, deep=True):
        return self.xgb_model.get_params(deep)

    def set_params(self, **parameters):
        self.xgb_model.set_params(**parameters)
        return self