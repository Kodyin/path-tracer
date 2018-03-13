// Based on sequential version path tracer by Kevin Beason
 
#include <iostream>
#include <stdio.h>
#include <ctime>
#include <cuda_runtime.h>
#include <vector_types.h>
#include "cutil_math.h" 
#include "timer.h"

#define width 1024  // screenwidth
#define height 768 // screenheight
#define samps 1000 // samples 


struct Ray { 
    float3 orig; // ray origin
    float3 dir;  // ray direction 
    __device__ Ray(float3 o_, float3 d_) : orig(o_), dir(d_) {} 
};

enum Refl_t { DIFF, SPEC, REFR };  

struct Sphere {

    float rad;            // radius 
    float3 pos, emi, col; // position, emission, colour 
    Refl_t refl;          // reflection type (e.g. diffuse)

    __device__ float intersect_sphere(const Ray &r) const { 
        // general sphere equation: x^2 + y^2 + z^2 = rad^2 
        // classic quadratic equation of form ax^2 + bx + c = 0
        float3 op = pos - r.orig;    // distance from ray.orig to center sphere 
        float t, epsilon = 0.0001f;  // epsilon required to prevent floating point precision artefacts
        float b = dot(op, r.dir);    // b in quadratic equation
        float disc = b*b - dot(op, op) + rad*rad;  // discriminant quadratic equation
        if (disc<0) return 0;       // if disc < 0, no real solution (we're not interested in complex roots) 
        else disc = sqrtf(disc);    // if disc >= 0, check for solutions using negative and positive discriminant
        return (t = b - disc)>epsilon ? t : ((t = b + disc)>epsilon ? t : 0); // pick closest point in front of ray origin
    }
};

// SCENE
// { float radius, { float3 position }, { float3 emission }, { float3 colour }, refl_type }
__constant__ Sphere spheres[] = {
    { 1e5f, { 1e5f + 1.0f, 40.8f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { 0.75f, 0.25f, 0.25f }, DIFF }, //Left 1e5f
    { 1e5f, { -1e5f + 99.0f, 40.8f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .25f, .25f, .75f }, DIFF }, //Right 
    { 1e5f, { 50.0f, 40.8f, 1e5f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Back 
    { 1e5f, { 50.0f, 40.8f, -1e5f + 600.0f }, { 0.0f, 0.0f, 0.0f }, { 0.00f, 0.00f, 0.00f }, DIFF }, //Front 
    { 1e5f, { 50.0f, -1e5f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Bottom 
    { 1e5f, { 50.0f, -1e5f + 81.6f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Top 
    { 16.5f, { 27.0f, 16.5f, 47.0f }, { 0.0f, 0.0f, 0.0f }, { 0.999f, 0.999f, 0.999f }, REFR }, // small sphere 1
    { 16.5f, { 73.0f, 16.5f, 78.0f }, { 0.0f, 0.0f, 0.0f }, { 0.999f, 0.999f, 0.999f }, SPEC }, // small sphere 2
    { 600.0f, { 50.0f, 681.6f - .27f, 81.6f }, { 12.0f, 12.0f, 12.0f }, { 0.0f, 0.0f, 0.0f }, DIFF }  // Light 12, 10 ,8
};

__device__ inline bool intersect_scene(const Ray &r, float &t, int &id){

    float n = sizeof(spheres) / sizeof(Sphere), d, inf = t = 1e20;  // t is distance to closest intersection
    for (int i = int(n); i--;)  // test all scene objects for intersection
        if ((d = spheres[i].intersect_sphere(r)) && d<t){  // if newly computed intersection distance d is smaller 
            t = d;  // keep track of distance along ray to closest intersection point 
            id = i; // and closest intersected object
        }
    return t<inf; // returns true if an intersection with the scene occurred, false when no hit
}

// random number generator 
__device__ static float getrandom(unsigned int *seed) {
    seed[0] = 36969 * ((seed[0]) & 65535) + ((seed[0]) >> 16);  // hash the seeds using bitwise AND and bitshifts
    seed[1] = 18000 * ((seed[1]) & 65535) + ((seed[1]) >> 16);

    unsigned int ires = ((seed[0]) << 16) + (seed[1]);

    // Convert to float
    union {
        float f;
        unsigned int ui;
    } res;

    res.ui = (ires & 0x007fffff) | 0x40000000;  // bitwise AND, bitwise OR
    return (res.f - 2.f) / 2.f;
}

// radiance function, the meat of path tracing 
// solves the rendering equation: 
// outgoing radiance (at a point) = emitted radiance + reflected radiance
// reflected radiance is sum (integral) of incoming radiance from all directions in hemisphere above point, 
// multiplied by reflectance function of material (BRDF) and cosine incident angle 


__device__ float3 radiance(Ray &r, unsigned int *ss){ // returns ray color

    float3 accucolor = make_float3(0.0f, 0.0f, 0.0f); // accumulates ray colour with each iteration through bounce loop
    float3 mask = make_float3(1.0f, 1.0f, 1.0f); 

    // ray bounce loop (no Russian Roulette used) 
    for (int bounces = 0; bounces < 5; bounces++){  // iteration up to 4 bounces (replaces recursion in CPU code)

    float t;           // distance to closest intersection 
    int id = 0;        // index of closest intersected sphere 

    float3 f;  // primitive colour
    float3 emit; // primitive emission colour
    float3 x; // intersection point
    float3 n; // normal
    float3 nl; // oriented normal
    float3 d; // ray direction of next path segment
    Refl_t refltype;
    // test ray for intersection with scene
    if (!intersect_scene(r, t, id))
     return make_float3(0.0f, 0.0f, 0.0f); // if miss, return black

    // we've got a hit!
    // compute hitpoint and normal
    const Sphere &obj = spheres[id];  // hitobject
    x = r.orig + r.dir*t;          // hitpoint 
    n = normalize(x - obj.pos);    // normal
    nl = dot(n, r.dir) < 0 ? n : n * -1; // front facing normal
    f = obj.col;   // object colour
    refltype = obj.refl;
    emit = obj.emi;
    // add emission of current sphere to accumulated colour
    // (first term in rendering equation sum) 
    accucolor += mask * emit;
  
    if (refltype == DIFF){
        // create 2 random numbers
        float r1 = 2 * M_PI * getrandom(ss);
        float r2 = getrandom(ss);
        float r2s = sqrtf(r2);

        // compute orthonormal coordinate frame uvw with hitpoint as origin 
        float3 w = nl;
        float3 u = normalize(cross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
        float3 v = cross(w, u);

        // compute cosine weighted random ray direction on hemisphere 
        d = normalize(u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrtf(1 - r2));

        // offset origin next path segment to prevent self intersection
        x += nl * 0.03;

        // multiply mask with colour of object
        mask *= f;
    }

    // ideal specular reflection (mirror) 
    else if (refltype == SPEC){

        // compute relfected ray direction according to Snell's law
        d = r.dir - 2.0f * n * dot(n, r.dir);

        // offset origin next path segment to prevent self intersection
        x += nl * 0.01f;

        // multiply mask with colour of object
        mask *= f;
    }

    // ideal refraction 
    if (refltype == REFR){

        bool into = dot(n, nl) > 0; // is ray entering or leaving refractive material?
        float nc = 1.0f;  // Index of Refraction air
        float nt = 1.5f;  // Index of Refraction glass/water
        float nnt = into ? nc / nt : nt / nc;  // IOR ratio of refractive materials
        float ddn = dot(r.dir, nl);
        float cos2t = 1.0f - nnt*nnt * (1.f - ddn*ddn);

    		if (cos2t < 0.0f) // total internal reflection 
    		{
      			d = reflect(r.dir, n); //d = r.dir - 2.0f * n * dot(n, r.dir);
      			x += nl * 0.01f;
    		}
    		else // cos2t > 0
    		{
      			// compute direction of transmission ray
      			float3 tdir = normalize(r.dir * nnt - n * ((into ? 1 : -1) * (ddn*nnt + sqrtf(cos2t))));

      			float R0 = (nt - nc)*(nt - nc) / (nt + nc)*(nt + nc);
      			float c = 1.f - (into ? -ddn : dot(tdir, n));
      			float Re = R0 + (1.f - R0) * c * c * c * c * c;
      			float Tr = 1 - Re; // Transmission
      			float P = .25f + .5f * Re;
      			float RP = Re / P;
      			float TP = Tr / (1.f - P);

      			// randomly choose reflection or transmission ray
      			if (getrandom(ss) < 0.25) // reflection ray
      			{
        				mask *= RP;
        				d = reflect(r.dir, n);
        				x += nl * 0.02f;
      			}
      			else // transmission ray
      			{
        				mask *= TP;
        				d = tdir; //r = Ray(x, tdir); 
        				x += nl * 0.0005f; // epsilon must be small to avoid artefacts
    		    }
    		}
    }
    // set up origin and direction of next path segment
    r.orig = x;
    r.dir = d;
    }
    return accucolor;
}


union Color  // 4 bytes = 4 chars = 1 float
{
  	float c;
  	uchar4 components;
};

__global__ void render_kernel(float3 *output){

   // assign a CUDA thread to every pixel (x,y)
   unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;   
   unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

   unsigned int i = (height - y - 1)*width + x; // index of current pixel 
   unsigned int ss[2] = {x,y};  // seeds for random number generator

    // generate ray directed at lower left corner of the screen
    Ray cam(make_float3(50, 52, 295.6), normalize(make_float3(0, -0.042612, -1))); // first hardcoded camera ray(origin, direction) 
    float3 cx = make_float3(width * .5135 / height, 0.0f, 0.0f); // ray direction offset in x direction
    float3 cy = normalize(cross(cx, cam.dir)) * .5135; // ray direction offset in y direction (.5135 is field of view angle)
    float3 r; // r is final pixel color       
    
    for (int sy=0; sy<2; sy++)     // 2x2 subpixel rows 
        for (int sx=0; sx<2; sx++){
            r = make_float3(0.0f); // reset r to zero for every pixel 
     
            for (int s = 0; s < samps; s++){  // samples per pixel
                // compute primary ray direction
                float r1 = 2*getrandom(ss), dx=r1<1 ? sqrt(r1)-1: 1-sqrt(2-r1); 
                float r2 = 2*getrandom(ss), dy=r2<1 ? sqrt(r2)-1: 1-sqrt(2-r2);
                float3 d = cam.dir + cx*(((sx + .5 + dx)*0.5 + x) / width - .5) + cy*(((sy + .5 + dy)*0.5 + y) / height - .5);

                // create primary ray, add incoming radiance to pixelcolor
                Ray ray = Ray(cam.orig + d * 140, normalize(d));
                r = r + radiance(ray, ss)*(1. / samps);
            } // Camera rays are pushed forward to start in interior 
            // write rgb value of pixel to image buffer on the GPU, clamp value to [0.0f, 1.0f] range
            output[i] += make_float3(clamp(r.x, 0.0f, 1.0f), clamp(r.y, 0.0f, 1.0f), clamp(r.z, 0.0f, 1.0f))*0.25;
        }
}

inline float clamp(float x){ return x < 0.0f ? 0.0f : x > 1.0f ? 1.0f : x; } 

// convert range [0,1] to int in range [0, 255] and gamma correction
inline int toInt(float x){ return int(pow(clamp(x), 1 / 2.2) * 255 + .5); }  

int main(){

    float3* output_h = new float3[width*height]; // pointer to memory for image on the host (system RAM)
    float3* output_d;    // pointer to memory for image on the device (GPU VRAM)

    std::clock_t start;
    long double duration;

    start = std::clock(); 

    cudaMalloc(&output_d, width * height * sizeof(float3));
    dim3 block(8, 8, 1);   
    dim3 grid(width / block.x, height / block.y, 1);

    printf("CUDA initialised.\nStart rendering...\n");

    // schedule threads on device and launch CUDA kernel from host
    render_kernel <<< grid, block >>>(output_d);  

    // copy results of computation from device back to host
    cudaMemcpy(output_h, output_d, width * height *sizeof(float3), cudaMemcpyDeviceToHost);  

    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

    fprintf (stdout, "Time to execute first add GPU reduction kernel: %Lg secs\n", duration);
    // free CUDA memory
    cudaFree(output_d);  
    printf("Done!\n");

    // Write image to PPM file, a very simple image file format
    FILE *f = fopen("smallptcuda.ppm", "w");          
    fprintf(f, "P3\n%d %d\n%d\n", width, height, 255);
    for (int i = 0; i < width*height; i++)  // loop over pixels, write RGB values
    fprintf(f, "%d %d %d ", toInt(output_h[i].x),
                            toInt(output_h[i].y),
                            toInt(output_h[i].z));

    printf("Saved image to 'smallptcuda.ppm'\n");

    delete[] output_h;
}



