"""DigitalHeat reference-house model: one scheduler, used by the year simulation and the live controller.

Modules
    config    every number, loaded from config/assumptions.yaml (value, unit, status, source)
    profile   hourly heat demand of the reference house from the Open-Meteo weather file
    physics   the tank balance (the only place energy moves)
    forecast  demand forecasts (perfect, archived Open-Meteo runs, persistence) and the heat reserve
    jobs      contracted batch jobs (arrival, work, completion window)
    policy    decide() and assign(): the scheduler
    sim       the 8,760-hour loop and its ledgers
    kpi       every key figure, computed from the ledgers only
    economics break-even, full cost, sizing
    export    writes the JSON files the dashboard reads
    live      replays a day against a real running process
"""
