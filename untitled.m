% =========================================================================
% DRISHTI DIGITAL TWIN - TELEMETRY POST-PROCESSING & VALIDATION
% Plots multi-channel simulated telemetry and checks operational limits
% =========================================================================

% 1. Extract Telemetry Signal Data from Workspace
try
    % Extract signals from 'out_engine_telemetry' bus structure
    time_vec = out_engine_telemetry.time;
    telemetry = out_engine_telemetry.signals;

    % Map individual signals from bus indices
    MAP_data = telemetry(1).values;
    CHT_data = telemetry(2).values;
    EGT_data = telemetry(3).values;
    RPM_data = telemetry(4).values;
    Vibe_data = telemetry(5).values;
catch
    error('Could not find out_engine_telemetry dataset. Ensure simulation has run (Ctrl+T).');
end

% 2. Rotax 914 UL Operational Safety Thresholds
MAP_limit = 39.9;    % Max continuous boost (inHg)
CHT_limit = 135.0;   % Max continuous CHT (°C)
EGT_limit = 880.0;   % Max continuous EGT (°C)
RPM_limit = 5800.0;  % Max continuous RPM

% 3. Plot Telemetry Channel Dashboard
figure('Name', 'DRISHTI Rotax 914 Digital Twin - Flight Telemetry', 'Color', [1 1 1]);

% Subplot 1: Engine Speed (RPM)
subplot(3, 2, 1);
plot(time_vec, RPM_data, 'b-', 'LineWidth', 1.5); hold on;
yline(RPM_limit, 'r--', 'Max Continuous (5800 RPM)', 'LineWidth', 1.2);
title('Engine Speed (RPM)'); xlabel('Time (s)'); ylabel('RPM'); grid on;

% Subplot 2: Manifold Absolute Pressure (MAP)
subplot(3, 2, 2);
plot(time_vec, MAP_data, 'm-', 'LineWidth', 1.5); hold on;
yline(MAP_limit, 'r--', 'Max Boost Limit (39.9 inHg)', 'LineWidth', 1.2);
title('Manifold Absolute Pressure (MAP)'); xlabel('Time (s)'); ylabel('inHg'); grid on;

% Subplot 3: Cylinder Head Temperature (CHT)
subplot(3, 2, 3);
plot(time_vec, CHT_data, 'r-', 'LineWidth', 1.5); hold on;
yline(CHT_limit, 'r--', 'Max CHT (135°C)', 'LineWidth', 1.2);
title('Cylinder Head Temperature (CHT)'); xlabel('Time (s)'); ylabel('°C'); grid on;

% Subplot 4: Exhaust Gas Temperature (EGT)
subplot(3, 2, 4);
plot(time_vec, EGT_data, 'k-', 'LineWidth', 1.5); hold on;
yline(EGT_limit, 'r--', 'Max EGT (880°C)', 'LineWidth', 1.2);
title('Exhaust Gas Temperature (EGT)'); xlabel('Time (s)'); ylabel('°C'); grid on;

% Subplot 5: Structural Vibration Level
subplot(3, 2, [5, 6]);
plot(time_vec, Vibe_data, 'g-', 'LineWidth', 1.2); hold on;
yline(2.5, 'r--', 'Warning Limit (2.5 g)', 'LineWidth', 1.2);
title('Engine Assembly Vibration (g)'); xlabel('Time (s)'); ylabel('Acceleration (g)'); grid on;

% 4. Automated Safety & Fault Detection Check
fprintf('\n================ TELEMETRY VALIDATION REPORT ================\n');
if max(RPM_data) > RPM_limit
    warning('OVERSPEED DETECTED: Peak RPM (%.1f) exceeded limit (%.1f RPM)!', max(RPM_data), RPM_limit);
end
if max(CHT_data) > CHT_limit
    warning('OVERHEAT DETECTED: Peak CHT (%.1f°C) exceeded limit (%.1f°C)!', max(CHT_data), CHT_limit);
end
if max(EGT_data) > EGT_limit
    warning('EXHAUST OVERHEAT DETECTED: Peak EGT (%.1f°C) exceeded limit (%.1f°C)!', max(EGT_data), EGT_limit);
end
if max(Vibe_data) > 2.5
    warning('HIGH VIBRATION DETECTED: Peak Vibration (%.2f g) indicates physical fault!', max(Vibe_data));
end
fprintf('Validation complete. All system variables processed successfully.\n');