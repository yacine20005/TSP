#ifndef SWARM_H
#define SWARM_H

#include "route.h"
#include "calc.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define SWARM_SIZE CYCLE_SIZE
#define MAX_VELOCITY_SIZE (CITY_SIZE * 4)
#define DEFAULT_EPSILON 1e-3f

/**
 * @brief Couple de permutation (a, b) dans une vitesse V = {(a, b), ...}
 */
typedef struct
{
    int a;
    int b;
} Permutation;

/**
 * @brief Vitesse V_k d'une particule : liste ordonnée de couples de permutations
 */
typedef struct
{
    Permutation perms[MAX_VELOCITY_SIZE];
    int size;
} Velocity;

/**
 * @brief Structure représentant l'essaim de particules selon le TP PSO
 *
 */
typedef struct
{
    int size;
    int max_iterations;

    float rho;
    float b1;
    float b2;

    float epsilon;
    Velocity velocities[CYCLE_SIZE];
    RouteCollection particles;

    RouteCollection pbest;
    float pbest_distances[CYCLE_SIZE];

    Route best_route;
    int best_route_index;
    float best_route_distance;
    
    Route prev_best_route;
    float prev_best_route_distance;
} Swarm;

/**
 * @brief Initialise une vitesse vide : V_i <- 0
 *
 * @return Velocity Vitesse avec size = 0
 */
Velocity init_velocity();

/**
 * @brief Initialise l'essaim selon l'algorithme du TP :
 * - Choisir k solutions aléatoires X_1, X_2, ..., X_k
 * - V_i <- 0 (i = 1, ..., k)
 * - Choisir k meilleures solutions P_1, ..., P_k
 * - Initialiser aléatoirement la meilleure solution de l'essaim P_g0
 * - Choisir rho entre 0 et 1
 *
 * @param swarm Pointeur vers l'essaim à initialiser
 * @param cities Route contenant la liste initiale des villes
 * @param num_particles Nombre de particules k
 * @param max_iterations Nombre maximal d'itérations N
 * @param epsilon Seuil de convergence
 * @return int 0 si succès, -1 si erreur
 */
int init_swarm(Swarm *swarm, Route *cities, int num_particles, int max_iterations, float epsilon);

/**
 * @brief Opération (V1 (+) V2) : Concaténation de deux vitesses
 *
 * Si les deux opérandes sont des vitesses, cette opération est la concaténation
 * de ses deux opérandes pour donner une vitesse.
 * Exemple : V1 = {(3,4)} et V2 = {(2,5)} => V1 (+) V2 = {(3,4), (2,5)}
 *
 * @param v1 Première vitesse
 * @param v2 Deuxième vitesse
 * @return Velocity La vitesse résultant de la concaténation V1 (+) V2
 */
Velocity add_velocities(Velocity v1, Velocity v2);

/**
 * @brief Opération (X (+) V) : Application d'une vitesse sur une position
 *
 * Si le premier opérande est une position et le deuxième est une vitesse,
 * cette opération est l'équivalent de permutations dans le vecteur de position.
 *
 * @param position La position actuelle X_k
 * @param velocity La vitesse V à appliquer
 * @return Route La nouvelle position X_k (+) V
 */
Route apply_velocity_to_position(Route position, Velocity velocity);

/**
 * @brief Opération (k (*) V) : Multiplication entre un réel k et une vitesse V
 *
 * Règles du TP :
 * - Si k < 0 : k <- -k et on se ramène aux cas suivants
 * - Si 0 < k < 1 : c <- floor(k * |V|), tronquer V de c éléments
 * - Si k est un entier : addition (concaténation) de la vitesse V k fois
 * - Si k > 1 (non entier) : séparer la partie entière de la partie réelle
 *   et se ramener aux cas précédents
 *
 * @param k Le coefficient réel (ex: rho, b1, b2)
 * @param v La vitesse V
 * @return Velocity La vitesse résultante k (*) V
 */
Velocity multiply_velocity(float k, Velocity v);

/**
 * @brief Opération (X1 (-) X2) : Différence entre deux positions
 *
 * Cette opération se fait entre deux positions pour donner la vitesse
 * faisant passer d'une position à l'autre.
 *
 * @param x1 Première position
 * @param x2 Deuxième position
 * @return Velocity La vitesse composée des permutations nécessaires
 */
Velocity subtract_positions(Route x1, Route x2);

/**
 * @brief Calcule la nouvelle vitesse V_{k+1} d'une particule selon l'équation du TP :
 *
 * V_{k+1} = (rho (*) V_k) (+) (b1 (*) (P_i (-) X_k)) (+) (b2 (*) (P_g (-) X_k))
 *
 * @param v_k Vitesse actuelle V_k
 * @param x_k Position actuelle X_k
 * @param p_i Meilleure position personnelle P_i
 * @param p_g Meilleure position de l'essaim P_g
 * @param rho Inertie
 * @param b1 Coefficient aléatoire b1
 * @param b2 Coefficient aléatoire b2
 * @return Velocity La nouvelle vitesse V_{k+1}
 */
Velocity compute_next_velocity(Velocity v_k, Route x_k, Route p_i, Route p_g, float rho, float b1, float b2);

/**
 * @brief Met à jour la meilleure solution de l'essaim P_g parmi P_1, P_2, ..., P_k
 *
 * @param swarm Pointeur vers l'essaim
 */
void update_global_best(Swarm *swarm);

/**
 * @brief Met à jour la meilleure solution personnelle P_i de chaque particule
 *
 * @param swarm Pointeur vers l'essaim
 */
void update_personal_bests(Swarm *swarm);

/**
 * @brief Exécute une itération de la boucle while du TP :
 * - Choisir b1, b2 aléatoirement
 * - P_g <- la meilleure solution entre P_1, P_2, ..., P_k
 * - Mettre à jour les vitesses V_i (i = 1, ..., k)
 * - Mettre à jour X_i = X_i (+) V_i (i = 1, ..., k)
 * - Mettre à jour la meilleure solution de chaque particule P_i (i = 1, ..., k)
 *
 * @param swarm Pointeur vers l'essaim
 * @return int 1 si la condition d'arrêt |P_g - P_g0| < epsilon est atteinte, 0 sinon, -1 si erreur
 */
int pso_step(Swarm *swarm);

/**
 * @brief Exécute la boucle complète while i < N avec test d'arrêt |P_g - P_g0| < epsilon
 *
 * @param swarm Pointeur vers l'essaim
 * @return Route La meilleure solution finale P_g
 */
Route pso_solve(Swarm *swarm);

#endif // SWARM_H
