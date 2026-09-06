@app.get("/")
def home(request: Request):
    surf = mission_surface(12000.0, 55.0)
    context = {
        "request": request,  # Pass request inside the context dictionary for modern Starlette/FastAPI
        "uav_id": "UAV-01 (Primary Testbed)",
        "flight_status": "IN FLIGHT",
        "rpm": surf["RPM"],
        "map": surf["MAP"],
        "cht": 85.2,
        "egt": 672.0,
        "fuel_flow": 38.1,
        "bus_voltage": 14.1,
        "alt_health": 95,
        "fuel_qty": 62,
        "engine_health": 93,
        "active_faults": 0,
        "predicted_rul": "2.8 h",
        "rul_confidence": "80%",
        "mission_completion": "78%",
        "altitude": 10000.0,
        "throttle": 55.0,
        "mission_time": 4.0,
    }
    # Pass request as the first argument, followed by the template name and context dictionary
    return templates.TemplateResponse(request, "dashboard.html", context)