class ApiKey:
    def __init__(self, _token):
        self._token = _token

    @property
    def token(self):
        return self._token

TgApiKey = ApiKey("7262257458:AAE1rE1qELroz1Iv59h_lBFktKE4qE88uys")