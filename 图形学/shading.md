Normals play an essential role in shading, where they are used to compute the brightness of objects.

Normals can be thought of as vectors with one caveat: **they do not transform the same way as vectors**. This is one of the main reasons we take the time to differentiate them. 


Right-hand rule: 
- Tangent, Bi-tangent, Normal. 
    ![](https://www.scratchapixel.com/images/geometry/normal.png)
- where Normal is usually aligned along y- or z-axis
    ![](https://www.scratchapixel.com/images/geometry/normal2.png)



If a group of photons hits an object, three things can happen: they can be either **absorbed**吸收, **reflected**反射, or **transmitted**透射. 

The **percentage** of photons reflected, absorbed, and transmitted varies from material to another and generally dictates how the object appears in the scene. 

Absorption process is responsible for the object's **color**. 

White light is made up of "red", "blue", and "green" photons. If a white light illuminates a red object, the absorption process filters out (or **absorbs**) the "green" and the "blue" photons. Because the object does not absorb the "red" photons, they are **reflected**. 

Each point on an illuminated area or object **emits / radiates / reflects** light rays in every direction. Only one beam from each point strikes the eye perpendicularly and can therefore be seen.