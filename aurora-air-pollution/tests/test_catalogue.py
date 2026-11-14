from airadv.catalogue import unavailable_fields


def test_hidden_analysis_not_silently_replaced_by_forecast():
    request = {"date": "2017-10-01", "type": "analysis", "leadtime_hour": "0",
               "time": ["00:00", "12:00"], "variable": ["particulate_matter_2.5um"]}
    rows = [{"date": ["2015-01-01/2022-12-31"], "type": ["forecast"],
             "leadtime_hour": ["0"], "time": ["00:00", "12:00"],
             "variable": ["particulate_matter_2.5um"]},
            {"date": ["2018-07-01/2022-12-31"], "type": ["analysis"],
             "time": ["00:00", "12:00"], "variable": ["particulate_matter_2.5um"]}]
    assert len(unavailable_fields(request, "sfc", rows)) == 2
    request["type"] = "forecast"
    assert unavailable_fields(request, "sfc", rows) == []
    request.update(date="2022-06-01", type="analysis")
    assert unavailable_fields(request, "sfc", rows) == []


def test_model_levels_cannot_satisfy_pressure_level_request():
    request = {"date": "2003-01-01", "time": ["00:00"], "variable": ["ozone"],
               "pressure_level": ["50", "1000"]}
    rows = [{"date": ["2003-01-01/2022-12-31"], "time": ["00:00"],
             "variable": ["ozone"], "model_level": ["50", "1000"]}]
    assert unavailable_fields(request, "plev", rows) == ["ozone@00:00"]
    rows[0]["pressure_level"] = rows[0].pop("model_level")
    assert unavailable_fields(request, "plev", rows) == []
