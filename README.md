# COMPASS

*Learning to Transfer School Attendance Zoning across Cities*

Designing school attendance zones often means starting a new round of parameter search and calibration for every city. COMPASS learns from that experience so it can be reused across cities.

The framework predicts **14 interpretable zoning-control parameters** from a city's spatial structure, school supply, student demand, demographic conditions, and inferred zoning difficulty. A mixed-integer linear programming (MILP) model then turns those parameters into an explicit zoning plan, accounting for enrollment balance, educational equity, school quality, travel distance, and capacity shortage. The output includes both the plan and the control parameters behind it, making the policy choices open to inspection and adjustment.

COMPASS combines **difficulty-aware parameter prediction, MILP, and Conditional Rescue**. Difficulty signals guide parameter learning and adjustment. When the initial solve needs further support, Rescue revises the configuration in a predefined order and re-solves the zoning problem.

In the paper's five-fold evaluation across **3,000 US cities**, COMPASS achieved a **99.80% solver-feasible success rate** and a **49.14x median runtime speedup** over repeating Oracle Search for each city.

## Repository contents

| File or directory | Contents |
| --- | --- |
| `Compass.py` | The complete COMPASS pipeline: difficulty-aware parameter prediction, parameter adjustment and calibration, MILP zoning, and ordered Conditional Rescue. Produces zoning assignments, maps, policy summaries, and solver diagnostics. |
| `Compass_Train.py` | Model training for COMPASS, including city-feature and structural-label construction, the zoning-difficulty classifier, and parameter prediction. Saves five-fold models, out-of-fold predictions, and full-data models. |
| `Training_Targets.csv` | Training supervision for 3,000 cities: the 14 search-derived zoning-control parameters, together with infeasible-attempt counts and rates used to construct the plan-generation difficulty labels. |
| `ZoneOptimization_CityData/` | The processed school-zoning dataset for 3,000 US cities, organized by city. Includes spatial boundaries and assignment units, adjacency graphs, elementary-school records, and the demographic, student-demand, capacity, and school-quality inputs used by the framework. |
