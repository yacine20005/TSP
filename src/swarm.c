#include "swarm.h"
#include <string.h>

/**
 * @brief Génère un nombre flottant aléatoire entre 0.0 et 1.0
 */
float random_float_0_1(void)
{
    return (float)rand() / (float)RAND_MAX;
}

Velocity init_velocity()
{
    Velocity v;
    v.size = 0;
    return v;
}

int init_swarm(Swarm *swarm, Route *cities, int num_particles, int max_iterations, float epsilon)
{
    if (swarm == NULL || cities == NULL || num_particles <= 0 || num_particles > CYCLE_SIZE)
        return -1;

    swarm->size = num_particles;
    swarm->max_iterations = max_iterations;
    swarm->epsilon = epsilon;

    swarm->rho = random_float_0_1();
    swarm->b1 = random_float_0_1();
    swarm->b2 = random_float_0_1();

    swarm->particles = init_route_collection();
    swarm->pbest = init_route_collection();

    // Choisir k solutions aléatoires X_1, ..., X_k
    // V_i <- 0 (i = 1, ..., k)
    // Initialiser P_1, ..., P_k aux positions initiales
    for (int i = 0; i < num_particles; i++)
    {
        Route random_route = generate_random_path(*cities, cities->size);
        add_route_to_collection(&swarm->particles, random_route);
        add_route_to_collection(&swarm->pbest, random_route);

        swarm->pbest_distances[i] = calc_distance_array(random_route);
        swarm->velocities[i] = init_velocity();
    }

    // Initialiser aléatoirement la meilleure solution de l'essaim P_g0
    int random_index = rand() % num_particles;
    swarm->prev_best_route = swarm->pbest.paths[random_index];
    swarm->prev_best_route_distance = swarm->pbest_distances[random_index];

    swarm->best_route = swarm->prev_best_route;
    swarm->best_route_index = random_index;
    swarm->best_route_distance = swarm->prev_best_route_distance;

    // Mettre à jour P_g avec la meilleure solution parmi P_1, ..., P_k
    update_global_best(swarm);

    return 0;
}

Velocity add_velocities(Velocity v1, Velocity v2)
{
    Velocity result = init_velocity();

    for (int i = 0; i < v1.size && result.size < MAX_VELOCITY_SIZE; i++)
    {
        result.perms[result.size++] = v1.perms[i];
    }

    for (int i = 0; i < v2.size && result.size < MAX_VELOCITY_SIZE; i++)
    {
        result.perms[result.size++] = v2.perms[i];
    }

    return result;
}

Route apply_velocity_to_position(Route position, Velocity velocity)
{
    Route new_position = position;

    for (int k = 0; k < velocity.size; k++)
    {
        int idx_a = velocity.perms[k].a;
        int idx_b = velocity.perms[k].b;

        if (idx_a >= 0 && idx_a < new_position.size &&
            idx_b >= 0 && idx_b < new_position.size)
        {
            City temp = new_position.cities[idx_a];
            new_position.cities[idx_a] = new_position.cities[idx_b];
            new_position.cities[idx_b] = temp;
        }
    }

    return new_position;
}

Velocity multiply_velocity(float k, Velocity v)
{
    Velocity result = init_velocity();

    if (v.size <= 0 || k == 0.0f)
        return result;

    // Si k < 0 : k <- -k et on se ramène aux cas précédents
    if (k < 0.0f)
        k = -k;

    // Si 0 < k < 1 : tronquer V de floor(k * |V|) éléments
    if (k < 1.0f)
    {
        int count = (int)floorf(k * (float)v.size);
        if (count > MAX_VELOCITY_SIZE)
            count = MAX_VELOCITY_SIZE;

        for (int i = 0; i < count; i++)
        {
            result.perms[result.size++] = v.perms[i];
        }
        return result;
    }

    // Si k >= 1 (entier ou réel > 1) :
    // On sépare la partie entière de la partie réelle et on se ramène aux cas précédents
    int int_part = (int)floorf(k);
    float frac_part = k - (float)int_part;

    // Partie entière : addition de vitesses (concaténation int_part fois)
    for (int i = 0; i < int_part; i++)
    {
        result = add_velocities(result, v);
    }

    // Partie réelle (si k n'est pas entier)
    if (frac_part > 0.0f)
    {
        Velocity frac_velocity = multiply_velocity(frac_part, v);
        result = add_velocities(result, frac_velocity);
    }

    return result;
}

Velocity subtract_positions(Route x1, Route x2)
{
    Velocity v = init_velocity();
    if (x1.size != x2.size)
        return v;

    Route temp = x1;

    // Trouver la suite de permutations faisant passer de x1 à x2
    for (int i = 0; i < temp.size; i++)
    {
        if (strcmp(temp.cities[i].name, x2.cities[i].name) != 0)
        {
            for (int j = i + 1; j < temp.size; j++)
            {
                if (strcmp(temp.cities[j].name, x2.cities[i].name) == 0)
                {
                    if (v.size < MAX_VELOCITY_SIZE)
                    {
                        v.perms[v.size].a = i;
                        v.perms[v.size].b = j;
                        v.size++;
                    }

                    City swap = temp.cities[i];
                    temp.cities[i] = temp.cities[j];
                    temp.cities[j] = swap;
                    break;
                }
            }
        }
    }

    return v;
}

Velocity compute_next_velocity(Velocity v_k, Route x_k, Route p_i, Route p_g, float rho, float b1, float b2)
{
    // V_{k+1} = (rho (*) V_k) (+) (b1 (*) (P_i (-) X_k)) (+) (b2 (*) (P_g (-) X_k))
    Velocity term_inertia = multiply_velocity(rho, v_k);

    Velocity diff_pbest = subtract_positions(x_k, p_i);
    Velocity term_cognitive = multiply_velocity(b1, diff_pbest);

    Velocity diff_gbest = subtract_positions(x_k, p_g);
    Velocity term_social = multiply_velocity(b2, diff_gbest);

    Velocity temp = add_velocities(term_inertia, term_cognitive);
    return add_velocities(temp, term_social);
}

void update_global_best(Swarm *swarm)
{
    if (swarm == NULL || swarm->size <= 0)
        return;

    for (int i = 0; i < swarm->size; i++)
    {
        if (swarm->pbest_distances[i] < swarm->best_route_distance)
        {
            swarm->best_route = swarm->pbest.paths[i];
            swarm->best_route_distance = swarm->pbest_distances[i];
            swarm->best_route_index = i;
        }
    }
}

void update_personal_bests(Swarm *swarm)
{
    if (swarm == NULL)
        return;

    for (int i = 0; i < swarm->size; i++)
    {
        float current_distance = calc_distance_array(swarm->particles.paths[i]);
        if (current_distance < swarm->pbest_distances[i])
        {
            swarm->pbest.paths[i] = swarm->particles.paths[i];
            swarm->pbest_distances[i] = current_distance;
        }
    }
}

int pso_step(Swarm *swarm)
{
    if (swarm == NULL || swarm->size <= 0)
        return -1;

    // Choisir b1, b2 aléatoirement
    swarm->b1 = random_float_0_1();
    swarm->b2 = random_float_0_1();

    // P_g <- la meilleure solution entre P_1, P_2, ..., P_k
    update_global_best(swarm);

    // Mettre à jour les vitesses V_i et les positions X_i (i = 1, ..., k)
    for (int i = 0; i < swarm->size; i++)
    {
        swarm->velocities[i] = compute_next_velocity(
            swarm->velocities[i],
            swarm->particles.paths[i],
            swarm->pbest.paths[i],
            swarm->best_route,
            swarm->rho,
            swarm->b1,
            swarm->b2
        );

        swarm->particles.paths[i] = apply_velocity_to_position(
            swarm->particles.paths[i],
            swarm->velocities[i]
        );
    }

    // Mettre à jour la meilleure solution de chaque particule P_i (i = 1, ..., k)
    update_personal_bests(swarm);
    update_global_best(swarm);

    // if |P_g - P_g0| < epsilon then Fin des calculs
    int converged = 0;
    if (fabsf(swarm->best_route_distance - swarm->prev_best_route_distance) < swarm->epsilon)
    {
        converged = 1;
    }

    // P_g0 <- P_g
    swarm->prev_best_route = swarm->best_route;
    swarm->prev_best_route_distance = swarm->best_route_distance;

    return converged;
}

Route pso_solve(Swarm *swarm)
{
    if (swarm == NULL)
        return init_route();

    int i = 0;
    while (i < swarm->max_iterations)
    {
        if (pso_step(swarm) == 1)
        {
            break;
        }
        i++;
    }

    return swarm->best_route;
}
