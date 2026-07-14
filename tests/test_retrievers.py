from datetime import date

from litetemp_rag_finance.retrievers.time_fusion import TimeFusionRetriever


class TestTimeFusion:
    def test_temporal_score_decays(self):
        ret = TimeFusionRetriever(None, None)
        score_near = ret._temporal_score(date(2024, 6, 1), date(2024, 7, 1))
        score_far = ret._temporal_score(date(2020, 1, 1), date(2024, 7, 1))
        assert score_near > score_far
