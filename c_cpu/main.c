#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <sys/timeb.h>

#define SIZE 250000

typedef struct particle{
    double x, y, z;
} particle;

particle randomize(particle k, double D, double time); /* to randomize the position of current
                                                        * particle via normal distribution random
                                                        * number generator */
int check_if_inside(particle k, double d, double r_rcv); // checks if particle k is inside receiver
double normal(double mean, double sigma); //normal dist random num generator
double distance_to_rcv(particle k, double d, double r_rcv);

int main() {

    particle particles[SIZE]; // position of particles (MMs)
    int flag[SIZE]; /* particle case
                     * 0  -> not reached
                     * 1  -> reached
                     * -1 -> out of boundry, cannot reach
                     * */

    double d = 4, r_rcv = 10;
    int count = 0; // #reached MMs
    int count_t = 0; // #reached MMs in one step time
    double D = 79.4; // Diffusion Coefficient

    double sim_t = 5.0; // simulator time
    double t = 0.001; // delta t
    double t_i = 0.001; //loop iterator
    double t1=0.0, t2=0.0;

//    printf("%f\n", (r_rcv/(r_rcv*d)) );
//    printf("%f\n", (d/( sqrt(4*M_PI*D*sim_t*sim_t*sim_t) )) );
//    printf("%f\n", exp(-d*d/(4*D*sim_t)) );
//
//    printf("%f\n", exp(-d*d/(4*D*sim_t)) * (d/( sqrt(4*M_PI*D*sim_t*sim_t*sim_t) )) * (r_rcv/(r_rcv*d)));

    for(int i=0; i<SIZE; i++){
        particles[i].x = 0;
        particles[i].y = 0;
        particles[i].z = 0;
    }

    for(int i=0; i<SIZE; i++)
        flag[i] = 0;

    struct timeb start, end;
    int diff;
    int i = 0;
    ftime(&start);


    while(t_i < sim_t){

        count_t = 0;

#pragma omp parallel for
        for(int i=0; i<SIZE; i++){

            //printf("%d /n", omp_get_thread_num());

            if(flag[i] == 0){
                particles[i] = randomize(particles[i], D, t);

                if( check_if_inside(particles[i], d, r_rcv) ){
                    count_t++;
                    count++;
                    flag[i] = 1;
                    t2 = t_i;
                    //printf("%f\n", t_i);
                    t1 = t_i;
                    //printf("%f\n", t_i);
                }

            }
        }

        if(count == SIZE)
            break;

        t_i += t;
    }

    ftime(&end);
    diff = (int) (1000.0 * (end.time - start.time)
                 + (end.millitm - start.millitm));

    printf("\nOperation took %u seconds\n", diff);

    printf("\n\n%d\n", count);

    return 0;
}

particle randomize(particle k, double D, double t){

    double sigma = sqrt(2 * D * t);

    //printf("%f\n", normal(0, sigma*sigma));

    k.x += normal(0, sigma*sigma);
    k.y += normal(0, sigma*sigma);
    k.z += normal(0, sigma*sigma);

    return k;
}

double normal(double mean, double sigma){

    double u, v, s;
    do {
        u = ((double)rand()/(double)RAND_MAX) * 2.0 - 1.0;
        v = ((double)rand()/(double)RAND_MAX) * 2.0 - 1.0;
        s = u * u + v * v;
    } while (s >= 1 || s == 0);

    double mult = sqrt(-2.0 * log(s) / s);

    //printf("%f\n", mean + sigma * u * mult);

    return mean + sigma * u * mult;

}

int check_if_inside(particle k, double d, double r_rcv){

    if (distance_to_rcv(k, d, r_rcv) <= r_rcv)
        return 1;

    return 0;
}

double distance_to_rcv(particle k, double d, double r_rcv){

    double dx = (k.x - (d + r_rcv));    dx *= dx;
    double dy = (k.y - 0);              dy *= dy;
    double dz = (k.z - 0);              dz *= dz;

    return sqrt(dx + dy + dz);
}
