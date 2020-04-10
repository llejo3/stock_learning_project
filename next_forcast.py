from trains.learning_n_mock_investment import LearningNMockInvestment
from params.global_params import GlobalParams


def main(corp_names: list = None, train_model: str = 'rnn', invest_only: bool = False) -> None:
    if corp_names is None:
        corp_names = ["에이치엘비", "동일철강", "키다리스튜디오"]

    params = GlobalParams('FORCAST', train_model=train_model)

    invests = LearningNMockInvestment(params)
    invests.train_n_invests_for_name(corp_names, invest_only)


def forcasts(corps_n_date: list) -> None:
    params = GlobalParams('FORCAST')
    invests = LearningNMockInvestment(params)
    invests.forcasts(corps_n_date)


if __name__ == '__main__':
    main(["중앙에너비스", "동신건설", "푸드웰", "대림제지", "삼아알미늄",
          "MH에탄올", "청보산업", "금호전기", "진로발효", "흥국화재",
          "피델릭스", "이스타코", "삼영화학공업", "에쓰씨엔지니어링", "리더스 기술투자", "한일화학", "PN풍년",  "LG전자"
          ], train_model='rnn')