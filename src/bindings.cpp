#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <glm/glm.hpp>
#include <vector>
#include <string>
#include <cmath>
#include "SpiceUsr.h"
namespace py = pybind11;

// --- COPY OF REQUIRED STRUCTS TO AVOID DEPENDENCY HELL ---
enum class MissionObjective { Flyby = 0, OrbitInsertion = 1, Impact = 2 };

struct NavTarget {
    int spiceID;
    float tofDays;
    MissionObjective objective;
    float targetAltKm;
};

enum class TriggerType { GET = 0, APSIS = 1, ALTITUDE = 2 };

struct BurnEntry {
    TriggerType trigger;
    double get_h = 0, get_m = 0, get_s = 0; 
    int apsisType = 0;              
    double targetAltKM = 0;         
    int altCondition = 0;           
    int altRefBodyID = 0;           
    double dvx = 0, dvy = 0, dvz = 0;       
    int refBodyID = 0;              
    bool isVNB = false;                 
    bool enabled = true;
};

struct SimulationSettings {
    std::string start_date;
    std::string end_date;
    std::vector<int> active_bodies;
    double step_size_sec = 600.0;
};

class SpacecraftWrapper {
public:
    std::string name;
    int initial_center_id;
    glm::dvec3 initial_pos;
    glm::dvec3 initial_vel;
    double epoch_et;
    double mission_epoch_et;
};

class NavigationPlanWrapper {
public:
    std::vector<NavTarget> targets;
    void add_target(int target_id, MissionObjective obj, float target_alt_km) {
        NavTarget t;
        t.spiceID = target_id;
        t.objective = obj;
        t.targetAltKm = target_alt_km;
        t.tofDays = 0.0f;
        targets.push_back(t);
    }
};

// --- C++ Original Math Logics from main.cpp ---
inline double StumpffC(double z) {
    if (z > 0) return (1.0 - std::cos(std::sqrt(z))) / z;
    if (z < 0) return (std::cosh(std::sqrt(-z)) - 1.0) / (-z);
    return 1.0 / 2.0;
}

inline double StumpffS(double z) {
    if (z > 0) return (std::sqrt(z) - std::sin(std::sqrt(z))) / std::pow(z, 1.5);
    if (z < 0) return (std::sinh(std::sqrt(-z)) - std::sqrt(-z)) / std::pow(-z, 1.5);
    return 1.0 / 6.0;
}

struct LambertResult {
    glm::dvec3 v1;
    glm::dvec3 v2;
    bool success;
};

inline LambertResult SolveLambert(const glm::dvec3& r1, const glm::dvec3& r2, double tof_sec, double mu, int centerBodyID = 0, bool prograde = true) {
    LambertResult result;
    result.success = false;
    double r1_mag = glm::length(r1);
    double r2_mag = glm::length(r2);
    glm::dvec3 r2_nudged = r2;
    double cos_dNu_check = glm::dot(r1, r2) / (r1_mag * r2_mag);
    if (std::abs(cos_dNu_check) > 0.9999) {
        r2_nudged += glm::dvec3(0.001, 0.001, 0.001);
        r2_mag = glm::length(r2_nudged);
    }
    double cos_dNu = glm::dot(r1, r2_nudged) / (r1_mag * r2_mag);
    cos_dNu = glm::clamp(cos_dNu, -1.0, 1.0);
    double dNu = std::acos(cos_dNu);
    if (!prograde) dNu = 2.0 * 3.14159265358979323846 - dNu;

    double A = std::sin(dNu) * std::sqrt((r1_mag * r2_mag) / (1.0 - std::cos(dNu)));
    if (A == 0.0) return result;

    double z_low = -4.0 * 3.141592653589 * 3.141592653589;
    double z_high = 4.0 * 3.141592653589 * 3.141592653589;
    double z = 0.0;
    double C = 0.0, S = 0.0, y = 0.0;
    bool converged = false;
    double TOL = 1e-6;

    for (int i = 0; i < 500; i++) {
        C = StumpffC(z);
        S = StumpffS(z);
        y = r1_mag + r2_mag + A * (z * S - 1.0) / std::sqrt(C);
        if (y < 0.0) { z_high = z; z = (z_low + z_high) / 2.0; continue; }
        double x = std::sqrt(y / C);
        double tof_calc = (std::pow(x, 3) * S + A * std::sqrt(y)) / std::sqrt(mu);
        if (std::abs(tof_calc - tof_sec) < TOL) { converged = true; break; }
        if (tof_calc <= tof_sec) z_low = z; else z_high = z; 
        z = (z_low + z_high) / 2.0;
    }
    if (!converged) return result;
    double f = 1.0 - y / r1_mag;
    double g = A * std::sqrt(y / mu);
    double g_dot = 1.0 - y / r2_mag;
    result.v1 = (r2 - r1 * f) / g;
    result.v2 = (r2 * g_dot - r1) / g;
    result.success = true;
    return result;
}

inline glm::dvec3 PropagateKepler(const glm::dvec3& r0, const glm::dvec3& v0, double dt, double mu, glm::dvec3& v_out) {
    double r0_mag = glm::length(r0);
    double v0_mag = glm::length(v0);
    double eps = (v0_mag * v0_mag) / 2.0 - mu / r0_mag;
    if (eps >= 0.0) { v_out = v0; return r0 + v0 * dt; }
    
    double a = -mu / (2.0 * eps);
    double n = std::sqrt(mu / (a * a * a));
    double rdotv = glm::dot(r0, v0);
    glm::dvec3 e_vec = ( (v0_mag * v0_mag - mu / r0_mag) * r0 - rdotv * v0 ) / mu;
    double e = glm::length(e_vec);
    if (e >= 1.0) { v_out = v0; return r0; }
    
    double sinE0 = 0.0;
    double cosE0 = 1.0;
    if (e > 1e-12) {
        sinE0 = rdotv / (e * std::sqrt(mu * a));
        cosE0 = (1.0 - r0_mag / a) / e;
    }
    double E0 = std::atan2(sinE0, cosE0);
    double M0 = E0 - e * std::sin(E0);
    double Mt = M0 + n * dt;
    
    double E = Mt;
    for (int i = 0; i < 20; ++i) {
        double f = E - e * std::sin(E) - Mt;
        double f_prime = 1.0 - e * std::cos(E);
        double dE = f / f_prime;
        E -= dE;
        if (std::abs(dE) < 1e-8) break;
    }
    
    double dfE = E - E0;
    double f_coeff = 1.0 - (a / r0_mag) * (1.0 - std::cos(dfE));
    double g_coeff = dt - std::sqrt(a * a * a / mu) * (dfE - std::sin(dfE));
    glm::dvec3 rt = f_coeff * r0 + g_coeff * v0;
    double rt_mag = glm::length(rt);
    double f_dot = -(std::sqrt(mu * a) / (rt_mag * r0_mag)) * std::sin(dfE);
    double g_dot = 1.0 - (a / rt_mag) * (1.0 - std::cos(dfE));
    v_out = f_dot * r0 + g_dot * v0;
    return rt;
}
double MeanToTrue(double M, double ecc) {
    if (ecc < 1.0) {
        double E = M;
        for (int i = 0; i < 15; i++) {
            double f = E - ecc * std::sin(E) - M;
            double df = 1.0 - ecc * std::cos(E);
            E -= f / df;
            if (std::abs(f) < 1e-11) break;
        }
        return 2.0 * std::atan2(std::sqrt(1.0 + ecc) * std::sin(E / 2.0), std::sqrt(1.0 - ecc) * std::cos(E / 2.0));
    } else {
        double F = M;
        for (int i = 0; i < 15; i++) {
            double f = ecc * std::sinh(F) - F - M;
            double df = ecc * std::cosh(F) - 1.0;
            F -= f / df;
            if (std::abs(f) < 1e-11) break;
        }
        return 2.0 * std::atan2(std::sqrt(ecc + 1.0) * std::sinh(F / 2.0), std::sqrt(ecc - 1.0) * std::cosh(F / 2.0));
    }
}

double TrueToMean(double nu, double ecc) {
    if (ecc < 1.0) {
        double E = 2.0 * std::atan(std::sqrt((1.0 - ecc) / (1.0 + ecc)) * std::tan(nu / 2.0));
        return E - ecc * std::sin(E);
    } else {
        double F = 2.0 * std::atanh(std::sqrt((ecc - 1.0) / (ecc + 1.0)) * std::tan(nu / 2.0));
        return ecc * std::sinh(F) - F;
    }
}

void SyncCartesianFromKeplerian(double sma, double ecc, double inc, double raan, double argp, double nu, double mu, double et,
                                glm::dvec3& pos, glm::dvec3& vel) {
    if (mu <= 0.0) return;
    SpiceDouble elts[8];
    double rp = (ecc < 1.0) ? (sma * (1.0 - ecc)) : (std::abs(sma) * (ecc - 1.0));
    elts[0] = rp;
    elts[1] = ecc;
    elts[2] = inc * 3.14159265358979 / 180.0;
    elts[3] = raan * 3.14159265358979 / 180.0;
    elts[4] = argp * 3.14159265358979 / 180.0;
    elts[5] = TrueToMean(nu * 3.14159265358979 / 180.0, ecc);
    elts[6] = et;
    elts[7] = mu;
    
    SpiceDouble state[6];
    conics_c(elts, et, state);
    pos = glm::dvec3(state[0], state[1], state[2]);
    vel = glm::dvec3(state[3], state[4], state[5]);
}

// Maps a barycenter SPICE ID (1-9) to the corresponding primary planet ID
// so that physical parameters (RADII, J2) can be queried correctly.
// e.g. 4 (Mars Barycenter) -> 499 (Mars), 5 -> 599 (Jupiter), etc.
// Non-barycenter IDs (>=100 or special) are returned unchanged.
inline int BarycenterToPlanetID(int id) {
    if (id >= 1 && id <= 9) return id * 100 + 99;
    return id;
}

double GetBodyGM(int spiceID) {
    SpiceInt n;
    SpiceDouble v[1];
    bodvcd_c(spiceID, "GM", 1, &n, v);
    if (failed_c()) {
        reset_c();
        // Try via the planet ID if this was a barycenter
        int planetID = BarycenterToPlanetID(spiceID);
        if (planetID != spiceID) {
            bodvcd_c(planetID, "GM", 1, &n, v);
            if (failed_c()) { reset_c(); return 0.0; }
        } else {
            return 0.0;
        }
    }
    return v[0];
}

double GetBodyRadius(int spiceID) {
    int queryID = BarycenterToPlanetID(spiceID); // always use planet, not barycenter
    SpiceInt n;
    SpiceDouble v[3];
    erract_c("SET", 0, "RETURN");
    errprt_c("SET", 0, "NONE");
    bodvcd_c(queryID, "RADII", 3, &n, v);
    double r = 0.0;
    if (!failed_c()) { r = v[0]; }
    reset_c();
    erract_c("SET", 0, "ABORT");
    errprt_c("SET", 0, "DEFAULT");
    return r;
}

struct CelestialBodyInfo {
    int SpiceID;
    double GM;
    double RadiusKM;
    double J2;
    std::string Name;
};

SpacecraftWrapper create_spacecraft_from_keplerian(
    std::string name,
    int center_id,
    double altitude_km,
    double eccentricity,
    double inclination,
    double raan,
    double arg_periapsis,
    double true_anomaly,
    std::string state_vector_epoch_utc,
    std::string mission_start_epoch_utc
) {
    SpacecraftWrapper sc;
    sc.name = name;
    sc.initial_center_id = center_id;
    
    double et_state, et_mission;
    str2et_c(state_vector_epoch_utc.c_str(), &et_state);
    str2et_c(mission_start_epoch_utc.c_str(), &et_mission);
    
    sc.epoch_et = et_state;
    sc.mission_epoch_et = et_mission;
    
    double mu = GetBodyGM(center_id);
    double radius = GetBodyRadius(center_id);
    double sma = altitude_km + radius;
    if (eccentricity >= 1.0) {
        sma = std::abs(altitude_km + radius);
    }
    
    glm::dvec3 pos, vel;
    SyncCartesianFromKeplerian(sma, eccentricity, inclination, raan, arg_periapsis, true_anomaly, mu, et_state, pos, vel);
    
    sc.initial_pos = pos;
    sc.initial_vel = vel;
    return sc;
}

std::vector<BurnEntry> calculate_navigation_plan(
    SpacecraftWrapper& sc,
    SimulationSettings& sim,
    NavigationPlanWrapper& nav_plan,
    double initial_delay_days
) {
    std::vector<BurnEntry> table;
    std::vector<NavTarget>& targets = nav_plan.targets;
    
    if (targets.empty()) return table;
    
    double start_et;
    str2et_c(sim.start_date.c_str(), &start_et);
    
    // CRITICAL FIX: Shift base_et by initial_delay_days BEFORE grid search!
    double base_et = start_et + initial_delay_days * 86400.0; 
    
    int centralBodyIdx = 10; 
    if (sc.initial_center_id == 399 && targets[0].spiceID == 301) centralBodyIdx = 399; 
    if (sc.initial_center_id == 301 && targets[0].spiceID == 399) centralBodyIdx = 399; 
    
    std::vector<CelestialBodyInfo> planets;
    for (int id : sim.active_bodies) {
        CelestialBodyInfo b;
        b.SpiceID = id;
        b.GM = GetBodyGM(id);
        b.RadiusKM = GetBodyRadius(id);
        b.J2 = 0.0;
        int j2QueryID = BarycenterToPlanetID(id); // use planet ID for physical params
        if (bodfnd_c(j2QueryID, "J2")) {
            SpiceInt n; SpiceDouble j2v[1];
            bodvcd_c(j2QueryID, "J2", 1, &n, j2v);
            b.J2 = j2v[0];
        }
        // If not in kernel, J2 stays 0 (acceptable for interplanetary transfers)
        if (b.J2 == 0.0 && id == 399) b.J2 = 0.00108263; // ESMAT.exe fallback for Earth
        
        // Approximate name
        SpiceChar name[32]; SpiceBoolean found;
        bodc2n_c(id, 32, name, &found);
        if (found) b.Name = name; else b.Name = std::to_string(id);
        planets.push_back(b);
    }

    double mu = GetBodyGM(centralBodyIdx);
    double sc_mu = GetBodyGM(sc.initial_center_id);
    
    double target_gm = GetBodyGM(targets[0].spiceID);
    double target_r = GetBodyRadius(targets[0].spiceID);
    
    double lt;
    double stateTgt[6];
    spkgeo_c(targets[0].spiceID, base_et, "J2000", centralBodyIdx, stateTgt, &lt);
    glm::dvec3 target_pos(stateTgt[0], stateTgt[1], stateTgt[2]);

    glm::dvec3 sc_pos = sc.initial_pos;
    glm::dvec3 sc_vel = sc.initial_vel;
    if (sc.initial_center_id != centralBodyIdx) {
        double scC[6]; spkgeo_c(sc.initial_center_id, base_et, "J2000", centralBodyIdx, scC, &lt);
        sc_pos += glm::dvec3(scC[0], scC[1], scC[2]);
        sc_vel += glm::dvec3(scC[3], scC[4], scC[5]);
    }

    double r1_dist = glm::length(sc_pos);
    double r2_dist = glm::length(target_pos);
    double a = (r1_dist + r2_dist) / 2.0;
    double T_h = 3.1415926535 * std::sqrt((a*a*a)/mu); 

    double T1 = 2.0 * 3.1415926535 * std::sqrt((r1_dist*r1_dist*r1_dist)/mu);
    double T2 = 2.0 * 3.1415926535 * std::sqrt((r2_dist*r2_dist*r2_dist)/mu);
    double diff = std::abs(1.0/T1 - 1.0/T2);
    double S = diff > 1e-12 ? (1.0 / diff) : T1; 

    double min_dv = 1e9;
    double best_dep = initial_delay_days;
    double best_tof = T_h / 86400.0;

    double dep_step = (S / 86400.0) / 40.0; 
    double tof_step = (T_h / 86400.0) / 40.0;
    if (dep_step < 0.1) dep_step = 0.1;
    if (tof_step < 0.1) tof_step = 0.1;

    double dep_max = (S / 86400.0) + dep_step; // One full synodic period

    if (r1_dist < 150000.0) {
        // Spacecraft is in a parking orbit: search over 2 orbital periods
        double v0_mag = glm::length(sc_vel);
        double eps = (v0_mag * v0_mag) / 2.0 - sc_mu / glm::length(sc_pos);
        if (eps < 0.0) {
            double sma_o = -sc_mu / (2.0 * eps);
            double T_orbit = 2.0 * 3.1415926535 * std::sqrt((sma_o*sma_o*sma_o)/sc_mu);
            dep_max = (T_orbit / 86400.0) * 2.0;
            dep_step = (T_orbit / 86400.0) / 100.0;
        } else {
            dep_max = 0.0;
        }
    }

    LambertResult best_lam;
    glm::dvec3 best_test_v;
    glm::dvec3 best_test_r;
    glm::dvec3 best_target_v;
    glm::dvec3 best_target_r;

    glm::dvec3 park_r = sc.initial_pos;
    glm::dvec3 park_v = sc.initial_vel;
    
    for (double dep = 0; dep <= dep_max; dep += dep_step) {
        double testET = base_et + dep * 86400.0;
        
        glm::dvec3 test_r = park_r;
        glm::dvec3 test_v = park_v;
        if (dep > 0.0) {
            test_r = PropagateKepler(park_r, park_v, dep * 86400.0, sc_mu, test_v);
        }

        glm::dvec3 planet_r(0.0);
        glm::dvec3 planet_v(0.0);
        if (sc.initial_center_id != centralBodyIdx) {
            double stateC[6];
            spkgeo_c(sc.initial_center_id, testET, "J2000", centralBodyIdx, stateC, &lt);
            planet_r = glm::dvec3(stateC[0], stateC[1], stateC[2]);
            planet_v = glm::dvec3(stateC[3], stateC[4], stateC[5]);
            test_r += planet_r;
            test_v += planet_v;
        }
        
        double hohmann_tof_days = T_h / 86400.0;
        // TOF search range and step are fully derived from Hohmann TOF
        double search_tof_min = hohmann_tof_days * 0.4;
        double search_tof_max = hohmann_tof_days * 1.5;
        
        // Exact dynamic TOF resolution from ESMAT.exe
        double current_tof_steps = (hohmann_tof_days < 10.0) ? 40.0 : 120.0;
        tof_step = (search_tof_max - search_tof_min) / current_tof_steps;

        for (double tof_d = search_tof_min; tof_d <= search_tof_max; tof_d += tof_step) {
            double arrET = testET + tof_d * 86400.0;
            double stateT[6];
            spkgeo_c(targets[0].spiceID, arrET, "J2000", centralBodyIdx, stateT, &lt);
            glm::dvec3 target_pos_center(stateT[0], stateT[1], stateT[2]);
            glm::dvec3 target_v(stateT[3], stateT[4], stateT[5]);
            
            LambertResult lam = SolveLambert(test_r, target_pos_center, tof_d * 86400.0, mu, centralBodyIdx, true);
            if (lam.success) {
                double current_dv = 0.0;
                
                if (sc.initial_center_id != centralBodyIdx) {
                    glm::dvec3 v_inf_vec = lam.v1 - planet_v;
                    double v_inf = glm::length(v_inf_vec);
                    glm::dvec3 r_rel = test_r - planet_r;
                    glm::dvec3 v_rel = test_v - planet_v;
                    double r_mag = glm::length(r_rel);
                    double v_req_at_r = std::sqrt(v_inf * v_inf + 2.0 * sc_mu / r_mag);
                    double speed_diff = std::abs(v_req_at_r - glm::length(v_rel));
                    // BUG FIX: Alignment cezasi kaldirildi.
                    // Interplanetary departure'da spacecraft park yorungesinde optimal
                    // fazi secebilir, dolayisiyla yalnizca speed_diff (gercek TMI DV) kullanilir.
                    current_dv = speed_diff;
                } else {
                    current_dv = glm::length(lam.v1 - test_v);
                }

                glm::dvec3 v_inf_vec_opt = lam.v2 - target_v;
                double v_inf_sq_opt = glm::dot(v_inf_vec_opt, v_inf_vec_opt);
                double r_peri_opt = target_r + targets[0].targetAltKm;
                double v_actual_opt = std::sqrt(v_inf_sq_opt + 2.0 * target_gm / r_peri_opt);
                double v_circ_opt   = std::sqrt(target_gm / r_peri_opt);
                double dv2 = std::abs(v_actual_opt - v_circ_opt);

                double total_score = current_dv;
                if (targets[0].objective != MissionObjective::Flyby) {
                    total_score += dv2;
                }
                
                total_score += (tof_d / 10.0) * 0.05;
                if (total_score < min_dv) {
                    min_dv = total_score;
                    best_dep = dep;
                    best_tof = tof_d;
                    best_lam = lam;
                    best_test_v = test_v;
                    best_test_r = test_r;
                    best_target_v = target_v;
                    best_target_r = target_pos_center;
                    py::print("[GRID-SEARCH] New Minima -> T_dep: ", dep, " d, TOF: ", tof_d, " d | dV_tot: ", min_dv, " km/s", py::arg("flush")=true);
                }
            }
        }
    }
    
    if (min_dv < 1e8 && sc.initial_center_id != centralBodyIdx) {
        // Optimize parking orbit phase for alignment with V_infinity
        double stC[6], lt2;
        spkgeo_c(sc.initial_center_id, base_et + best_dep * 86400.0, "J2000", centralBodyIdx, stC, &lt2);
        glm::dvec3 planet_v(stC[3], stC[4], stC[5]);
        glm::dvec3 v_inf_vec = best_lam.v1 - planet_v;
        
        glm::dvec3 r_at_dep = park_r;
        glm::dvec3 v_at_dep = park_v;
        if (best_dep > 0.0) {
            r_at_dep = PropagateKepler(park_r, park_v, best_dep * 86400.0, sc_mu, v_at_dep);
        }
        
        double v0_mag = glm::length(v_at_dep);
        double eps = (v0_mag * v0_mag) / 2.0 - sc_mu / glm::length(r_at_dep);
        double T_orbit = 86400.0;
        if (eps < 0.0) {
            double sma_o = -sc_mu / (2.0 * eps);
            T_orbit = 2.0 * 3.1415926535 * std::sqrt((sma_o*sma_o*sma_o)/sc_mu);
        }
        
        double v_inf_mag = glm::length(v_inf_vec);
        double e_hyp = 1.0 + glm::length(r_at_dep) * v_inf_mag * v_inf_mag / sc_mu;
        
        double best_phase_dt = 0.0;
        double max_align = -2.0;
        for (double dt = 0; dt < T_orbit; dt += T_orbit / 100.0) {
            glm::dvec3 r_test, v_test;
            r_test = PropagateKepler(r_at_dep, v_at_dep, dt, sc_mu, v_test);
            glm::dvec3 V_p_dir = glm::normalize(v_inf_vec / v_inf_mag + (1.0 / e_hyp) * glm::normalize(r_test));
            double align = glm::dot(V_p_dir, glm::normalize(v_test));
            if (align > max_align) {
                max_align = align;
                best_phase_dt = dt;
            }
        }
        
        double fine_start = std::max(0.0, best_phase_dt - T_orbit / 50.0);
        double fine_end = best_phase_dt + T_orbit / 50.0;
        for (double dt = fine_start; dt <= fine_end; dt += T_orbit / 5000.0) {
            glm::dvec3 r_test, v_test;
            r_test = PropagateKepler(r_at_dep, v_at_dep, dt, sc_mu, v_test);
            glm::dvec3 V_p_dir = glm::normalize(v_inf_vec / v_inf_mag + (1.0 / e_hyp) * glm::normalize(r_test));
            double align = glm::dot(V_p_dir, glm::normalize(v_test));
            if (align > max_align) {
                max_align = align;
                best_phase_dt = dt;
            }
        }
        
        best_dep += best_phase_dt / 86400.0;
    }

    // BUILD MISSION TABLE 
    if (best_dep > 0.001) {
        BurnEntry wait1;
        wait1.trigger = TriggerType::GET;
        double sec = best_dep * 86400.0;
        wait1.get_h = std::floor(sec / 3600.0);
        wait1.get_m = std::floor((sec - wait1.get_h * 3600.0) / 60.0);
        wait1.get_s = sec - wait1.get_h * 3600.0 - wait1.get_m * 60.0;
        wait1.dvx = 0; wait1.dvy = 0; wait1.dvz = 0;
        wait1.isVNB = false;
        wait1.refBodyID = sc.initial_center_id;
        table.push_back(wait1);
    }
    
    BurnEntry tmi;
    tmi.trigger = TriggerType::GET; 
    // initial_delay_days is already baked into base_et!
    // But we need to add it to the final GET output so it matches the mission epoch!
    double total_sec = (initial_delay_days + best_dep) * 86400.0;
    tmi.get_h = std::floor(total_sec / 3600.0);
    tmi.get_m = std::floor((total_sec - tmi.get_h * 3600.0) / 60.0);
    tmi.get_s = total_sec - tmi.get_h * 3600.0 - tmi.get_m * 60.0;
    
    glm::dvec3 dv_vec;
    bool tmi_isVNB = false;
    int tmi_refBodyID = centralBodyIdx;
    // Hoisted for LOI calculation (scope outside if block)
    LambertResult final_lam; final_lam.success = false;
    glm::dvec3 loi_target_v(0.0, 0.0, 0.0);
    double actual_peri_v = 0.0; // Hoisted for LOI calculation

    if (min_dv < 1e8) {
        double current_t = base_et + best_dep * 86400.0;
        
        // REVERT BUG FIX: Must cast to float to exactly match ESMAT.exe (which uses float in NavTarget)
        targets[0].tofDays = (float)best_tof;
        double tof_sec = targets[0].tofDays * 86400.0;
        
        // Exact N-Body Phase shift for the parking orbit
        glm::dvec3 current_r = sc.initial_pos;
        glm::dvec3 current_v = sc.initial_vel;
        if (best_dep > 0.0) {
            double wait_sec = best_dep * 86400.0;
            double t_current = base_et;
            
            // Central body for J2
            double cBody_J2 = 0.0;
            double cBody_Radius = 1.0;
            std::string cBody_Name = "EARTH";
            for (auto& p : planets) {
                if (p.SpiceID == sc.initial_center_id) {
                    cBody_J2 = p.J2;
                    cBody_Radius = p.RadiusKM;
                    cBody_Name = p.Name;
                    break;
                }
            }
            double mu_c = GetBodyGM(sc.initial_center_id);

            double h_wait_base = 10.0; // Precise steps for parking orbit
            int steps_wait = (int)std::ceil(wait_sec / h_wait_base);
            double h_wait = wait_sec / steps_wait;

            for (int s = 0; s < steps_wait; ++s) {
                auto get_acc_park = [&](glm::dvec3 p, double et) {
                    double rm = glm::length(p);
                    glm::dvec3 a = -mu_c * p / (rm*rm*rm);
                    if (cBody_J2 > 1e-9) {
                        double R_mat[3][3]; std::string iau = "IAU_" + cBody_Name;
                        for(auto &c: iau) c=toupper(c); pxform_c(iau.c_str(), "J2000", et, R_mat);
                        glm::dmat3 rot = glm::dmat3(R_mat[0][0],R_mat[1][0],R_mat[2][0],R_mat[0][1],R_mat[1][1],R_mat[2][1],R_mat[0][2],R_mat[1][2],R_mat[2][2]);
                        glm::dvec3 pl = glm::transpose(rot) * p; double r2=rm*rm; double r5=r2*r2*rm;
                        double j2f = -1.5*cBody_J2*mu_c*cBody_Radius*cBody_Radius/r5;
                        a += rot * glm::dvec3(j2f*pl.x*(1.0-5.0*pl.z*pl.z/r2), j2f*pl.y*(1.0-5.0*pl.z*pl.z/r2), j2f*pl.z*(3.0-5.0*pl.z*pl.z/r2));
                    }
                    for (auto& b : planets) {
                        if (b.SpiceID == sc.initial_center_id) continue;
                        double stB[6], local_lt; spkgeo_c(b.SpiceID, et, "J2000", sc.initial_center_id, stB, &local_lt);
                        glm::dvec3 rb(stB[0], stB[1], stB[2]);
                        glm::dvec3 r_rel = p - rb;
                        double d_mag = glm::length(r_rel), rb_mag = glm::length(rb);
                        if (d_mag > 1.0 && rb_mag > 1.0)
                            a += -b.GM * (r_rel/(d_mag*d_mag*d_mag) + rb/(rb_mag*rb_mag*rb_mag));
                    }
                    return a;
                };
                
                glm::dvec3 k1v=get_acc_park(current_r, t_current), k1r=current_v;
                glm::dvec3 k2v=get_acc_park(current_r+k1r*(h_wait/2.0), t_current+h_wait/2.0), k2r=current_v+k1v*(h_wait/2.0);
                glm::dvec3 k3v=get_acc_park(current_r+k2r*(h_wait/2.0), t_current+h_wait/2.0), k3r=current_v+k2v*(h_wait/2.0);
                glm::dvec3 k4v=get_acc_park(current_r+k3r*h_wait, t_current+h_wait), k4r=current_v+k3v*h_wait;
                current_v += (h_wait/6.0)*(k1v+2.0*k2v+2.0*k3v+k4v);
                current_r += (h_wait/6.0)*(k1r+2.0*k2r+2.0*k3r+k4r);
                t_current += h_wait;
            }
        }
        
        // CRITICAL: Convert from parking orbit frame (sc.initial_center_id)
        // to central body frame (centralBodyIdx) before Lambert and Virtual Pilot.
        // For interplanetary (e.g., Earth→Mars), centralBodyIdx=10 (Sun),
        // but dep wait was integrated in Earth-centered frame.
        if (sc.initial_center_id != centralBodyIdx) {
            double stC[6], lt_c;
            spkgeo_c(sc.initial_center_id, current_t, "J2000", centralBodyIdx, stC, &lt_c);
            current_r += glm::dvec3(stC[0], stC[1], stC[2]);
            current_v += glm::dvec3(stC[3], stC[4], stC[5]);
        }

        // Re-solve lambert with the exact Phase-Shifted state for inertial Delta-V baseline
        glm::dvec3 target_pos_center = best_target_r;
        glm::dvec3 target_v = best_target_v;
        // Re-query target state at arrival time for the final lambert (more accurate)
        double arr_et_final = current_t + tof_sec;
        {
            double stTfinal[6], lt_f;
            spkgeo_c(targets[0].spiceID, arr_et_final, "J2000", centralBodyIdx, stTfinal, &lt_f);
            target_pos_center = glm::dvec3(stTfinal[0], stTfinal[1], stTfinal[2]);
            target_v = glm::dvec3(stTfinal[3], stTfinal[4], stTfinal[5]);
        }
        LambertResult final_lam_inner = SolveLambert(current_r, target_pos_center, tof_sec, mu, centralBodyIdx, true);
        if (!final_lam_inner.success) final_lam_inner = best_lam; // fallback
        final_lam = final_lam_inner;  // expose to outer scope for LOI
        loi_target_v = target_v;      // expose arrival velocity to outer scope
        
        double target_gm = GetBodyGM(targets[0].spiceID);
        double target_radius = GetBodyRadius(targets[0].spiceID);
        
        int vnbRefSpice = sc.initial_center_id;
        glm::dvec3 v_rel_vp = current_v;
        glm::dvec3 r_rel_vp = current_r;
        if (vnbRefSpice != centralBodyIdx) {
            double stC[6], lt;
            spkgeo_c(vnbRefSpice, current_t, "J2000", centralBodyIdx, stC, &lt);
            r_rel_vp = current_r - glm::dvec3(stC[0], stC[1], stC[2]);
            v_rel_vp = current_v - glm::dvec3(stC[3], stC[4], stC[5]);
        }
        if (glm::length(r_rel_vp) < 1.0) { r_rel_vp = current_r; v_rel_vp = current_v; }
        
        glm::dvec3 V = glm::normalize(v_rel_vp);
        glm::dvec3 cross_rv = glm::cross(r_rel_vp, v_rel_vp);
        glm::dvec3 N = (glm::length(cross_rv) > 1e-10) ? glm::normalize(cross_rv) : glm::dvec3(0, 0, 1);
        glm::dvec3 B = glm::cross(V, N);

        glm::dvec3 dv_inertial;
        if (sc.initial_center_id != centralBodyIdx) {
            // Interplanetary: We must escape the parking orbit's gravity well (Oberth effect).
            // final_lam.v1 is the heliocentric velocity required.
            double stC[6], lt;
            spkgeo_c(sc.initial_center_id, current_t, "J2000", centralBodyIdx, stC, &lt);
            glm::dvec3 v_planet(stC[3], stC[4], stC[5]);
            
            glm::dvec3 v_inf = final_lam.v1 - v_planet;
            double v_inf_mag = glm::length(v_inf);
            double v_esc = std::sqrt(2.0 * sc_mu / glm::length(r_rel_vp));
            double v_p = std::sqrt(v_inf_mag * v_inf_mag + v_esc * v_esc);
            
            // Exact patched-conic required velocity (including out-of-plane and beta angle bend)
            double e_hyp = 1.0 + glm::length(r_rel_vp) * v_inf_mag * v_inf_mag / sc_mu;
            glm::dvec3 V_p_dir = glm::normalize(v_inf / v_inf_mag + (1.0 / e_hyp) * glm::normalize(r_rel_vp));
            
            glm::dvec3 required_v = V_p_dir * v_p;
            dv_inertial = required_v - v_rel_vp;
        } else {
            // Local transfer (e.g., Earth -> Moon)
            dv_inertial = final_lam.v1 - current_v;
        }
        
        double dv_v = glm::dot(dv_inertial, V);
        double dv_n = glm::dot(dv_inertial, N);
        double dv_b = glm::dot(dv_inertial, B);

        printf("[NAV-DBG] Target pos (CB frame): X=%d Y=%d Z=%d km\n", (int)target_pos_center.x, (int)target_pos_center.y, (int)target_pos_center.z);
        printf("[NAV-DBG] Dep pos (CB frame): X=%d Y=%d Z=%d km | |v|=%f\n", (int)current_r.x, (int)current_r.y, (int)current_r.z, glm::length(current_v));

        double j2 = 0.0;
        int j2CentralID = BarycenterToPlanetID(centralBodyIdx);
        if (bodfnd_c(j2CentralID, "J2")) {
            SpiceInt n;
            bodvcd_c(j2CentralID, "J2", 1, &n, &j2);
        }
        // If not in kernel, J2 stays 0
        
        double r_peri_target = target_radius + targets[0].targetAltKm;
        
        // Hoist J2 constants for Central Body
        double cBody_J2 = 0.0;
        double cBody_Radius = 1.0;
        std::string cBody_Name = "EARTH";
        for (auto& b : planets) {
            if (b.SpiceID == centralBodyIdx) {
                cBody_J2 = b.J2;
                cBody_Radius = b.RadiusKM;
                cBody_Name = b.Name;
                break;
            }
        }
        
        double dep_radius = (sc.initial_center_id != centralBodyIdx) ? GetBodyRadius(sc.initial_center_id) : 0.0;

        auto runVirtualFlight = [&](double dvv, double dvn, double dvb, double& out_v) {
            glm::dvec3 v_start = current_v + (V * dvv + N * dvn + B * dvb);
            glm::dvec3 r = current_r; glm::dvec3 v = v_start;
            double t = current_t;
            
            // Adaptive timestep for interplanetary missions, static 10s for local (e.g. Moon) missions
            double h_base = (centralBodyIdx == 10) ? 3600.0 : 10.0;
            
            double elapsed_t = 0.0;
            double min_dist = 1e18; 
            bool isImpactMode = (targets[0].objective == MissionObjective::Impact);
            double flight_duration = tof_sec + 86400.0 * 2.0; // 2 days padding for delayed arrivals
            
            while (elapsed_t < flight_duration) {
                double stT[6], lt; spkgeo_c(targets[0].spiceID, t, "J2000", centralBodyIdx, stT, &lt);
                glm::dvec3 r_tgt(stT[0], stT[1], stT[2]);
                glm::dvec3 v_tgt(stT[3], stT[4], stT[5]);
                double d_to_target = glm::length(r - r_tgt);
                
                if (isImpactMode && d_to_target <= target_radius) {
                    min_dist = d_to_target;
                    out_v = glm::length(v - v_tgt);
                    break;
                }
                
                double actual_h = h_base;
                if (centralBodyIdx == 10) {
                    // Time-based Departure (deterministic)
                    if (sc.initial_center_id != centralBodyIdx) {
                        if (elapsed_t < 86400.0 * 1.0) actual_h = std::min(actual_h, 10.0);
                        else if (elapsed_t < 86400.0 * 5.0) actual_h = std::min(actual_h, 60.0);
                        else if (elapsed_t < 86400.0 * 20.0) actual_h = std::min(actual_h, 600.0);
                    }
                    
                    // Time-based Arrival (deterministic)
                    double time_to_arrival = std::abs(tof_sec - elapsed_t);
                    if (time_to_arrival < 86400.0 * 1.0) actual_h = std::min(actual_h, 10.0);
                    else if (time_to_arrival < 86400.0 * 5.0) actual_h = std::min(actual_h, 60.0);
                    else if (time_to_arrival < 86400.0 * 20.0) actual_h = std::min(actual_h, 600.0);
                }
                
                if (isImpactMode && d_to_target < target_radius * 5.0)
                    actual_h = std::min(actual_h, 1.0);
                
                if (elapsed_t + actual_h > flight_duration) actual_h = flight_duration - elapsed_t;
                if (actual_h < 1e-6) break;

                auto get_acc = [&](glm::dvec3 p, double et) {
                    double rm = glm::length(p);
                    glm::dvec3 a = -mu * p / (rm*rm*rm);
                    
                    for (auto& b : planets) {
                        if (b.SpiceID == centralBodyIdx) continue;
                        double stB[6], local_lt; spkgeo_c(b.SpiceID, et, "J2000", centralBodyIdx, stB, &local_lt);
                        glm::dvec3 rb(stB[0], stB[1], stB[2]);
                        glm::dvec3 r_rel = p - rb;
                        double d_mag = glm::length(r_rel), rb_mag = glm::length(rb);
                        
                        if (d_mag > 1.0 && rb_mag > 1.0)
                            a += -b.GM * (r_rel/(d_mag*d_mag*d_mag) + rb/(rb_mag*rb_mag*rb_mag));
                    }
                    
                    // J2 Perturbation for Central Body using hoisted constants
                    if (cBody_J2 > 1e-9) {
                        double R_mat[3][3]; std::string iau = "IAU_" + cBody_Name;
                        for(auto &c: iau) c=toupper(c); pxform_c(iau.c_str(), "J2000", et, R_mat);
                        glm::dmat3 rot = glm::dmat3(R_mat[0][0],R_mat[1][0],R_mat[2][0],R_mat[0][1],R_mat[1][1],R_mat[2][1],R_mat[0][2],R_mat[1][2],R_mat[2][2]);
                        glm::dvec3 pl = glm::transpose(rot) * p; double r2 = rm * rm; double r5 = r2 * r2 * rm;
                        double j2f = -1.5 * cBody_J2 * mu * cBody_Radius * cBody_Radius / r5;
                        a += rot * glm::dvec3(j2f * pl.x * (1.0 - 5.0 * pl.z * pl.z / r2), 
                                              j2f * pl.y * (1.0 - 5.0 * pl.z * pl.z / r2), 
                                              j2f * pl.z * (3.0 - 5.0 * pl.z * pl.z / r2));
                    }
                    
                    return a;
                };

                glm::dvec3 r_pre = r;
                glm::dvec3 r_tgt_pre = r_tgt;

                glm::dvec3 k1v=get_acc(r,t), k1r=v;
                glm::dvec3 k2v=get_acc(r+k1r*(actual_h/2.0),t+actual_h/2.0), k2r=v+k1v*(actual_h/2.0);
                glm::dvec3 k3v=get_acc(r+k2r*(actual_h/2.0),t+actual_h/2.0), k3r=v+k2v*(actual_h/2.0);
                glm::dvec3 k4v=get_acc(r+k3r*actual_h,t+actual_h), k4r=v+k3v*actual_h;
                v += (actual_h/6.0)*(k1v+2.0*k2v+2.0*k3v+k4v);
                r += (actual_h/6.0)*(k1r+2.0*k2r+2.0*k3r+k4r);
                t += actual_h; elapsed_t += actual_h;

                double stT_post[6], lt_post; spkgeo_c(targets[0].spiceID, t, "J2000", centralBodyIdx, stT_post, &lt_post);
                glm::dvec3 r_tgt_post(stT_post[0], stT_post[1], stT_post[2]);
                glm::dvec3 v_tgt_post(stT_post[3], stT_post[4], stT_post[5]);

                glm::dvec3 p0 = r_pre - r_tgt_pre;
                glm::dvec3 p1 = r - r_tgt_post;
                glm::dvec3 v_seg = p1 - p0;

                double len_sq = glm::dot(v_seg, v_seg);
                double frac = (len_sq > 1e-12) ? std::clamp(-glm::dot(p0, v_seg) / len_sq, 0.0, 1.0) : 0.0;
                glm::dvec3 proj = p0 + frac * v_seg;

                double min_seg_dist = glm::length(proj);
                if (min_seg_dist < min_dist) {
                    min_dist = min_seg_dist;
                    glm::dvec3 v_tgt_interp = v_tgt + frac * (v_tgt_post - v_tgt);
                    glm::dvec3 v_interp = (r_pre + frac * (r - r_pre)); // approximate v is not needed here, we just use v_end or interpolate if we wanted. But out_v is just v - v_tgt.
                    // Let's just use v_post for out_v to be safe
                    out_v = glm::length(v - v_tgt_post);
                }
            }
            return min_dist;
        };

        double trash_v;
        double last_dist = 0;
        int max_iters = (sc.initial_center_id != centralBodyIdx) ? 150 : 50;
        
        for (int iter = 0; iter < max_iters; ++iter) {
            double d0 = runVirtualFlight(dv_v, dv_n, dv_b, actual_peri_v);
            last_dist = d0;
            double err = d0 - r_peri_target;
            
            if (targets[0].objective == MissionObjective::Impact && d0 <= r_peri_target) {
                py::print("[PILOT] Iter", iter, ": IMPACT at", (int)d0, "km from center - surface confirmed.", py::arg("flush")=true);
                break;
            }
            if (std::abs(err) < 0.1) break;

            double eps = 1e-2;
            double ddv = (runVirtualFlight(dv_v+eps,dv_n,dv_b,trash_v) - d0)/eps;
            double ddn = (runVirtualFlight(dv_v,dv_n+eps,dv_b,trash_v) - d0)/eps;
            double ddb = (runVirtualFlight(dv_v,dv_n,dv_b+eps,trash_v) - d0)/eps;

            double grad_mag = ddv*ddv + ddn*ddn + ddb*ddb;
            if (grad_mag > 1e-18) {
                double step = err / grad_mag;
                
                double step_v = step * ddv;
                double step_n = step * ddn;
                double step_b = step * ddb;
                double step_mag = std::sqrt(step_v*step_v + step_n*step_n + step_b*step_b);

                double max_adj = (sc.initial_center_id != centralBodyIdx && centralBodyIdx == 10) ? 2.5 : 0.5; 
                double relax_factor = (sc.initial_center_id != centralBodyIdx && centralBodyIdx == 10) ? 1.0 : 0.8;

                if (step_mag > max_adj) {
                    double scale = max_adj / step_mag;
                    step_v *= scale;
                    step_n *= scale;
                    step_b *= scale;
                }

                double current_relax = relax_factor;
                bool step_accepted = false;
                
                for (int ls = 0; ls < 6; ++ls) {
                    double try_v = dv_v - step_v * current_relax;
                    double try_n = dv_n - step_n * current_relax;
                    double try_b = dv_b - step_b * current_relax;
                    
                    double try_d = runVirtualFlight(try_v, try_n, try_b, trash_v);
                    double try_err = std::abs(try_d - r_peri_target);
                    
                    if (try_err < std::abs(err)) {
                        dv_v = try_v;
                        dv_n = try_n;
                        dv_b = try_b;
                        step_accepted = true;
                        break;
                    }
                    current_relax *= 0.5;
                }
                
                if (!step_accepted) {
                    py::print("[PILOT] Iter", iter, ": Line search exhausted. Local minimum reached.", py::arg("flush")=true);
                    break;
                }
            }
            py::print("[PILOT] Iter", iter, ": Periapsis=", (int)d0, "km (Err:", (int)err, "km)", py::arg("flush")=true);
        }
        py::print("[NAV] Virtual Pilot Converged at", (int)last_dist, "km radius (altitude:", (int)(last_dist - target_radius), "km).", py::arg("flush")=true);
        
        // Final Output as VNB
        // VNB convention: X=Velocity(prograde), Y=Normal(orbit-normal), Z=Binormal(radial)
        // ESMAT.exe mapping: dvx=V(prograde), dvy=B(binormal/radial), dvz=N(orbit-normal)
        tmi_isVNB = true;
        tmi_refBodyID = vnbRefSpice;
        dv_vec = glm::dvec3(dv_v, dv_b, dv_n); // Match ESMAT.exe mapping (v, b, n)
    }
    
    tmi.dvx = dv_vec.x;
    tmi.dvy = dv_vec.y;
    tmi.dvz = dv_vec.z;
    tmi.isVNB = tmi_isVNB;
    tmi.refBodyID = tmi_refBodyID;
    table.push_back(tmi);
    
    // Arrival
    if (targets[0].objective == MissionObjective::OrbitInsertion) {
        // Dynamic SOI: no hardcoded values.
        // centralBodyIdx is already set correctly (399 for Earth-Moon, 10 for interplanetary).
        // parentGM = GM of the body the target orbits = cruise center body.
        double parentGM = GetBodyGM(centralBodyIdx);
        double target_soi = -1.0;
        
        if (parentGM > 0.0) {
            // Get target position relative to its parent (centralBodyIdx) via SPICE
            double stTgt[6], lt_tgt;
            spkgeo_c(targets[0].spiceID, base_et, "J2000", centralBodyIdx, stTgt, &lt_tgt);
            double d_p = glm::length(glm::dvec3(stTgt[0], stTgt[1], stTgt[2]));
            
            // Hill Sphere / SOI: r_SOI = d * (GM_target / GM_parent)^(2/5)
            if (d_p > 1.0) {
                target_soi = d_p * std::pow(target_gm / parentGM, 0.4);
                py::print("[SOI] Target:", targets[0].spiceID, "| d_p:", (long long)d_p, "km | GM_parent:", (long long)parentGM, "km3/s2 | SOI:", (long long)target_soi, "km", py::arg("flush")=true);
            }
        }
        
        if (target_soi > 0) {
            BurnEntry soi_wait;
            soi_wait.trigger = TriggerType::ALTITUDE;
            soi_wait.altCondition = 1; // <=
            soi_wait.altRefBodyID = targets[0].spiceID;
            
            double target_radius = GetBodyRadius(targets[0].spiceID);
            double safe_altitude = (target_soi * 0.1) - target_radius;
            if (safe_altitude < 500.0) safe_altitude = 500.0;
            
            soi_wait.targetAltKM = safe_altitude; 
            soi_wait.refBodyID = 0; // Coast
            soi_wait.dvx = 0; soi_wait.dvy = 0; soi_wait.dvz = 0;
            table.push_back(soi_wait);
        } else {
            // Fallback to Apsis
            BurnEntry arr_wait;
            arr_wait.trigger = TriggerType::APSIS;
            arr_wait.apsisType = 1; // Periapsis
            arr_wait.refBodyID = targets[0].spiceID;
            table.push_back(arr_wait);
        }
        
        BurnEntry loi;
        loi.trigger = TriggerType::APSIS;
        loi.apsisType = 1; 
        
        // LOI DV: BUG FIX 2 (from ESMAT.exe) - Use actual periapsis velocity from Virtual Pilot for precise insertion
        double r_peri_opt   = target_r + targets[0].targetAltKm;
        double v_circ_opt   = std::sqrt(target_gm / r_peri_opt);
        double dv_loi = std::max(0.0, actual_peri_v - v_circ_opt);
        
        loi.dvx = -dv_loi; loi.dvy = 0; loi.dvz = 0; 
        loi.isVNB = true;
        loi.refBodyID = targets[0].spiceID;
        table.push_back(loi);
    } else if (targets[0].objective == MissionObjective::Flyby) {
        BurnEntry flyby_wait;
        flyby_wait.trigger = TriggerType::APSIS;
        flyby_wait.apsisType = 1; 
        flyby_wait.refBodyID = targets[0].spiceID;
        table.push_back(flyby_wait);
    }

    return table;
}

PYBIND11_MODULE(core, m) {
    m.doc() = "ESMAT Core Python Bindings";

    // Prevent SPICE from crashing the Python process on errors
    erract_c("SET", 0, (SpiceChar*)"RETURN");

    m.def("load_kernel", [](const std::string& path) { furnsh_c(path.c_str()); }, "Load a SPICE kernel");
    m.def("unload_kernel", [](const std::string& path) { unload_c(path.c_str()); }, "Unload a SPICE kernel");
    m.def("str2et", [](const std::string& time_str) { double et = 0.0; str2et_c(time_str.c_str(), &et); return et; }, "Convert UTC time string to ET");

    // Body name/code lookup — requires a loaded PCK/TPC kernel
    m.def("body_name", [](int id) -> std::string {
        SpiceChar name[64];
        SpiceBoolean found;
        bodc2n_c(id, 64, name, &found);
        if (found) return std::string(name);
        return "";
    }, py::arg("id"), "Get SPICE body name from integer ID. Returns empty string if not found.");

    m.def("body_code", [](const std::string& name) -> int {
        SpiceInt code;
        SpiceBoolean found;
        bodn2c_c(name.c_str(), &code, &found);
        if (found) return (int)code;
        return -1;
    }, py::arg("name"), "Get SPICE body ID from name string. Returns -1 if not found.");

    py::enum_<MissionObjective>(m, "MissionObjective")
        .value("Flyby", MissionObjective::Flyby)
        .value("OrbitInsertion", MissionObjective::OrbitInsertion)
        .value("Impact", MissionObjective::Impact)
        .export_values();

    py::enum_<TriggerType>(m, "TriggerType")
        .value("GET", TriggerType::GET)
        .value("APSIS", TriggerType::APSIS)
        .value("ALTITUDE", TriggerType::ALTITUDE)
        .export_values();

    py::class_<NavTarget>(m, "NavTarget")
        .def(py::init<>())
        .def_readwrite("spiceID", &NavTarget::spiceID)
        .def_readwrite("tofDays", &NavTarget::tofDays)
        .def_readwrite("objective", &NavTarget::objective)
        .def_readwrite("targetAltKm", &NavTarget::targetAltKm);

    py::class_<BurnEntry>(m, "BurnEntry")
        .def(py::init<>())
        .def_readwrite("trigger", &BurnEntry::trigger)
        .def_readwrite("get_h", &BurnEntry::get_h)
        .def_readwrite("get_m", &BurnEntry::get_m)
        .def_readwrite("get_s", &BurnEntry::get_s)
        .def_readwrite("apsisType", &BurnEntry::apsisType)
        .def_readwrite("targetAltKM", &BurnEntry::targetAltKM)
        .def_readwrite("altCondition", &BurnEntry::altCondition)
        .def_readwrite("altRefBodyID", &BurnEntry::altRefBodyID)
        .def_readwrite("dvx", &BurnEntry::dvx)
        .def_readwrite("dvy", &BurnEntry::dvy)
        .def_readwrite("dvz", &BurnEntry::dvz)
        .def_readwrite("isVNB", &BurnEntry::isVNB)
        .def_readwrite("refBodyID", &BurnEntry::refBodyID)
        .def("__repr__", [](const BurnEntry &b) {
            std::string s = "<BurnEntry trigger=";
            if (b.trigger == TriggerType::GET) {
                s += "Time(GET) params=" + std::to_string((int)b.get_h) + ":" + std::to_string((int)b.get_m) + ":" + std::to_string(b.get_s);
            } else if (b.trigger == TriggerType::APSIS) {
                s += "Apsis params=" + std::string(b.apsisType == 0 ? "Apoapsis" : "Periapsis");
            } else if (b.trigger == TriggerType::ALTITUDE) {
                const char* ops[] = {"<", "<=", ">=", ">"};
                s += "Altitude params=" + std::string(b.altCondition >= 0 && b.altCondition < 4 ? ops[b.altCondition] : "<=") + " " + std::to_string(b.targetAltKM) + " km";
            }
            s += " dV=(" + std::to_string(b.dvx) + "," + std::to_string(b.dvy) + "," + std::to_string(b.dvz) + ") ref=" + std::to_string(b.refBodyID) + ">";
            return s;
        });
        
    py::class_<SimulationSettings>(m, "SimulationSettings")
        .def(py::init<>())
        .def_readwrite("start_date", &SimulationSettings::start_date)
        .def_readwrite("end_date", &SimulationSettings::end_date)
        .def_readwrite("active_bodies", &SimulationSettings::active_bodies)
        .def_readwrite("step_size_sec", &SimulationSettings::step_size_sec);

    py::class_<SpacecraftWrapper>(m, "Spacecraft")
        .def(py::init<>())
        .def_readwrite("name", &SpacecraftWrapper::name)
        .def_readwrite("initial_center_id", &SpacecraftWrapper::initial_center_id)
        .def_property("initial_pos", 
            [](SpacecraftWrapper& sc) { return std::vector<double>{sc.initial_pos.x, sc.initial_pos.y, sc.initial_pos.z}; },
            [](SpacecraftWrapper& sc, std::vector<double> v) { sc.initial_pos = glm::dvec3(v[0], v[1], v[2]); })
        .def_property("initial_vel", 
            [](SpacecraftWrapper& sc) { return std::vector<double>{sc.initial_vel.x, sc.initial_vel.y, sc.initial_vel.z}; },
            [](SpacecraftWrapper& sc, std::vector<double> v) { sc.initial_vel = glm::dvec3(v[0], v[1], v[2]); })
        .def_readwrite("epoch", &SpacecraftWrapper::epoch_et)
        .def_readwrite("mission_epoch", &SpacecraftWrapper::mission_epoch_et)
        .def_static("from_keplerian", &create_spacecraft_from_keplerian,
            py::arg("name"), py::arg("center_id"), py::arg("altitude_km"), py::arg("eccentricity"), 
            py::arg("inclination"), py::arg("raan"), py::arg("arg_periapsis"), py::arg("true_anomaly"), 
            py::arg("state_vector_epoch_utc"), py::arg("mission_start_epoch_utc"));
            
    py::class_<NavigationPlanWrapper>(m, "NavigationPlan")
        .def(py::init<>())
        .def("add_target", &NavigationPlanWrapper::add_target, py::arg("target_id"), py::arg("objective"), py::arg("target_alt_km"));

    m.def("calculate_navigation_plan", &calculate_navigation_plan, 
        py::arg("spacecraft"), py::arg("simulation"), py::arg("targets"),
        py::arg("initial_delay_days") = 0.0);
}
