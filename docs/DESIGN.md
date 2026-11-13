# Design document

Status: this document grows phase by phase. Sections are added when the part they describe
is designed, so everything written here matches code that exists.

## 1. What is monitored, and why these signals

The conveyor is modelled as a fixed structure 5 km long (configurable) with 10 sensor
stations (configurable) at fixed kilometre positions. Each station reports five signals
every 5 seconds.

| Signal | Unit | Why it is monitored |
|--------|------|---------------------|
| Temperature | °C | Failing idler bearings and slipping drives turn mechanical energy into heat. Heat is a late but reliable sign, and an overheated idler is a fire risk on a belt carrying coal or ore dust. |
| Vibration | mm/s RMS | Bearing wear shows in vibration before it shows in temperature. It is the earliest of the five signals for idler and pulley faults. |
| Belt speed | m/s | Belt speed falling below drive speed, or fluctuating, means the belt is slipping on the drive pulley. |
| Belt load | % of rated | Load explains the other signals: more material means more current and more heat. Without it, a loaded belt looks like a faulty one. Sustained overload is also a failure cause in itself. |
| Motor current | A | Current is the drive's effort. Rising current at constant load means added friction (seized idlers, misalignment, slip). |

### Operating limits

One table of limits is defined in `backend/config.py` and used by the simulator, the
backend and the predictor.

| Signal | Typical normal | Warning limit | Failure limit | Fails by |
|--------|----------------|---------------|---------------|----------|
| Temperature | 35–60 °C | 75 °C | 95 °C | rising |
| Vibration | 1.5–3.5 mm/s | 7.1 mm/s | 11.0 mm/s | rising |
| Belt speed | 4.0 m/s nominal | 3.6 m/s | 3.2 m/s | falling |
| Belt load | 50–90 % | 100 % | 120 % | rising |
| Motor current | 180–300 A | 340 A | 400 A | rising |

Where the numbers come from:

- The vibration limits follow the scale of the ISO 10816 vibration-severity zones, where
  7.1 and 11 mm/s RMS are zone boundaries for large machines. They are used here as
  plausible magnitudes, not as a standards assessment of any real machine.
- The other limits are round numbers chosen to be plausible for a large belt drive. They
  are assumptions. On a real installation they would come from the equipment
  manufacturer's data and the site's failure history.

"Failure limit" is the level at which the simulator declares the component failed, and the
level the predictor extrapolates trends to.

## 2. Assumptions made where the brief was unclear

| # | Assumption | Reason |
|---|------------|--------|
| A1 | Every station reports all five signals, including motor current. | The brief lists the five signals per reading. On a real conveyor current is measured only at drives. A multi-kilometre belt often has more than one drive, but not one per station. The simulator keeps the brief's uniform schema; the design treats current and speed as "the drive conditions seen at this station". |
| A2 | Station positions are evenly spaced, at the centre of equal segments (0.25, 0.75, … 4.75 km by default). | The brief gives no layout. Even spacing is the simplest layout that covers the belt. |
| A3 | The belt loop is twice the conveyor length plus 30 m of pulley wrap (10 030 m by default). | The belt runs out on the carry side and back on the return side. The exact figure only matters for the vision registry, and it is configurable. |
| A4 | Nominal belt speed is 4.0 m/s, so one lap takes about 42 minutes. | Within the usual range for long overland conveyors. It sets how often the camera sees the same piece of belt. |
| A5 | Timestamps are UTC, ISO 8601. | Avoids daylight-saving and time-zone ambiguity between simulator, backend and dashboard. |
