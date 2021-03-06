
## Reshape in numpy


```python
import numpy as np

# to keep the example easy to view, we use 10 examples of pictures.  Each picture is of size 4x4 pixels, and each pixel contains 3 (r,g,b) values
# we first fill an array with sequential 480 entries (10*4*4*3) and then reshape it into an 10x4x4x3 array
a=np.arange(10*4*4*3).reshape(10, 4, 4, 3)
print(a.shape)
print(a)

# we can reshape arrays using .reshape method
# array axis are numbered from 0 and accessed through .shape method
# if a specific axis is specified to .reshape, that axis is preserved and the rest of the axis' are reshaped/flattened
# a -1 argument tells numpy to figure out the dimensions of reshaped axis 

# flatten the innermost axis (the r,g,b values), which are already flat, so this operation does nothing
aflat=a.reshape(a.shape[0],a.shape[1],a.shape[2],-1)
print(aflat.shape)
print(aflat)

# flatten the innermost two axis (r,g,b values in each pixel row). 4x3 gets flattened to 12 color values
aflat=a.reshape(a.shape[0],a.shape[1],-1)
print(aflat.shape)
print(aflat)

# flatten the innermost three axis (r,g,b values in each pixel row, reading left to right and top to bottom). 4x4x3 gets flattened to 48 values.  this operation flattens each individual image
aflat=a.reshape(a.shape[0],-1)
print(aflat.shape)
print(aflat)

# at this point, the rows have 'examples' (the training or test cases) and columns have the 'features' (the color values).  to get the features in rows and examples in columns, we transpose the matrix using the .T method
aflatt=aflat.T
print(aflatt.shape)
print(aflatt)

# fun exercise
# to create random pixel noise to test your trained network, try the following
# x_test=np.random.randint(255,size=(64*64*3,209))
# print(x_test.shape)
# print(x_test)
```

    (10, 4, 4, 3)
    [[[[  0   1   2]
       [  3   4   5]
       [  6   7   8]
       [  9  10  11]]
    
      [[ 12  13  14]
       [ 15  16  17]
       [ 18  19  20]
       [ 21  22  23]]
    
      [[ 24  25  26]
       [ 27  28  29]
       [ 30  31  32]
       [ 33  34  35]]
    
      [[ 36  37  38]
       [ 39  40  41]
       [ 42  43  44]
       [ 45  46  47]]]
    
    
     [[[ 48  49  50]
       [ 51  52  53]
       [ 54  55  56]
       [ 57  58  59]]
    
      [[ 60  61  62]
       [ 63  64  65]
       [ 66  67  68]
       [ 69  70  71]]
    
      [[ 72  73  74]
       [ 75  76  77]
       [ 78  79  80]
       [ 81  82  83]]
    
      [[ 84  85  86]
       [ 87  88  89]
       [ 90  91  92]
       [ 93  94  95]]]
    
    
     [[[ 96  97  98]
       [ 99 100 101]
       [102 103 104]
       [105 106 107]]
    
      [[108 109 110]
       [111 112 113]
       [114 115 116]
       [117 118 119]]
    
      [[120 121 122]
       [123 124 125]
       [126 127 128]
       [129 130 131]]
    
      [[132 133 134]
       [135 136 137]
       [138 139 140]
       [141 142 143]]]
    
    
     [[[144 145 146]
       [147 148 149]
       [150 151 152]
       [153 154 155]]
    
      [[156 157 158]
       [159 160 161]
       [162 163 164]
       [165 166 167]]
    
      [[168 169 170]
       [171 172 173]
       [174 175 176]
       [177 178 179]]
    
      [[180 181 182]
       [183 184 185]
       [186 187 188]
       [189 190 191]]]
    
    
     [[[192 193 194]
       [195 196 197]
       [198 199 200]
       [201 202 203]]
    
      [[204 205 206]
       [207 208 209]
       [210 211 212]
       [213 214 215]]
    
      [[216 217 218]
       [219 220 221]
       [222 223 224]
       [225 226 227]]
    
      [[228 229 230]
       [231 232 233]
       [234 235 236]
       [237 238 239]]]
    
    
     [[[240 241 242]
       [243 244 245]
       [246 247 248]
       [249 250 251]]
    
      [[252 253 254]
       [255 256 257]
       [258 259 260]
       [261 262 263]]
    
      [[264 265 266]
       [267 268 269]
       [270 271 272]
       [273 274 275]]
    
      [[276 277 278]
       [279 280 281]
       [282 283 284]
       [285 286 287]]]
    
    
     [[[288 289 290]
       [291 292 293]
       [294 295 296]
       [297 298 299]]
    
      [[300 301 302]
       [303 304 305]
       [306 307 308]
       [309 310 311]]
    
      [[312 313 314]
       [315 316 317]
       [318 319 320]
       [321 322 323]]
    
      [[324 325 326]
       [327 328 329]
       [330 331 332]
       [333 334 335]]]
    
    
     [[[336 337 338]
       [339 340 341]
       [342 343 344]
       [345 346 347]]
    
      [[348 349 350]
       [351 352 353]
       [354 355 356]
       [357 358 359]]
    
      [[360 361 362]
       [363 364 365]
       [366 367 368]
       [369 370 371]]
    
      [[372 373 374]
       [375 376 377]
       [378 379 380]
       [381 382 383]]]
    
    
     [[[384 385 386]
       [387 388 389]
       [390 391 392]
       [393 394 395]]
    
      [[396 397 398]
       [399 400 401]
       [402 403 404]
       [405 406 407]]
    
      [[408 409 410]
       [411 412 413]
       [414 415 416]
       [417 418 419]]
    
      [[420 421 422]
       [423 424 425]
       [426 427 428]
       [429 430 431]]]
    
    
     [[[432 433 434]
       [435 436 437]
       [438 439 440]
       [441 442 443]]
    
      [[444 445 446]
       [447 448 449]
       [450 451 452]
       [453 454 455]]
    
      [[456 457 458]
       [459 460 461]
       [462 463 464]
       [465 466 467]]
    
      [[468 469 470]
       [471 472 473]
       [474 475 476]
       [477 478 479]]]]
    (10, 4, 4, 3)
    [[[[  0   1   2]
       [  3   4   5]
       [  6   7   8]
       [  9  10  11]]
    
      [[ 12  13  14]
       [ 15  16  17]
       [ 18  19  20]
       [ 21  22  23]]
    
      [[ 24  25  26]
       [ 27  28  29]
       [ 30  31  32]
       [ 33  34  35]]
    
      [[ 36  37  38]
       [ 39  40  41]
       [ 42  43  44]
       [ 45  46  47]]]
    
    
     [[[ 48  49  50]
       [ 51  52  53]
       [ 54  55  56]
       [ 57  58  59]]
    
      [[ 60  61  62]
       [ 63  64  65]
       [ 66  67  68]
       [ 69  70  71]]
    
      [[ 72  73  74]
       [ 75  76  77]
       [ 78  79  80]
       [ 81  82  83]]
    
      [[ 84  85  86]
       [ 87  88  89]
       [ 90  91  92]
       [ 93  94  95]]]
    
    
     [[[ 96  97  98]
       [ 99 100 101]
       [102 103 104]
       [105 106 107]]
    
      [[108 109 110]
       [111 112 113]
       [114 115 116]
       [117 118 119]]
    
      [[120 121 122]
       [123 124 125]
       [126 127 128]
       [129 130 131]]
    
      [[132 133 134]
       [135 136 137]
       [138 139 140]
       [141 142 143]]]
    
    
     [[[144 145 146]
       [147 148 149]
       [150 151 152]
       [153 154 155]]
    
      [[156 157 158]
       [159 160 161]
       [162 163 164]
       [165 166 167]]
    
      [[168 169 170]
       [171 172 173]
       [174 175 176]
       [177 178 179]]
    
      [[180 181 182]
       [183 184 185]
       [186 187 188]
       [189 190 191]]]
    
    
     [[[192 193 194]
       [195 196 197]
       [198 199 200]
       [201 202 203]]
    
      [[204 205 206]
       [207 208 209]
       [210 211 212]
       [213 214 215]]
    
      [[216 217 218]
       [219 220 221]
       [222 223 224]
       [225 226 227]]
    
      [[228 229 230]
       [231 232 233]
       [234 235 236]
       [237 238 239]]]
    
    
     [[[240 241 242]
       [243 244 245]
       [246 247 248]
       [249 250 251]]
    
      [[252 253 254]
       [255 256 257]
       [258 259 260]
       [261 262 263]]
    
      [[264 265 266]
       [267 268 269]
       [270 271 272]
       [273 274 275]]
    
      [[276 277 278]
       [279 280 281]
       [282 283 284]
       [285 286 287]]]
    
    
     [[[288 289 290]
       [291 292 293]
       [294 295 296]
       [297 298 299]]
    
      [[300 301 302]
       [303 304 305]
       [306 307 308]
       [309 310 311]]
    
      [[312 313 314]
       [315 316 317]
       [318 319 320]
       [321 322 323]]
    
      [[324 325 326]
       [327 328 329]
       [330 331 332]
       [333 334 335]]]
    
    
     [[[336 337 338]
       [339 340 341]
       [342 343 344]
       [345 346 347]]
    
      [[348 349 350]
       [351 352 353]
       [354 355 356]
       [357 358 359]]
    
      [[360 361 362]
       [363 364 365]
       [366 367 368]
       [369 370 371]]
    
      [[372 373 374]
       [375 376 377]
       [378 379 380]
       [381 382 383]]]
    
    
     [[[384 385 386]
       [387 388 389]
       [390 391 392]
       [393 394 395]]
    
      [[396 397 398]
       [399 400 401]
       [402 403 404]
       [405 406 407]]
    
      [[408 409 410]
       [411 412 413]
       [414 415 416]
       [417 418 419]]
    
      [[420 421 422]
       [423 424 425]
       [426 427 428]
       [429 430 431]]]
    
    
     [[[432 433 434]
       [435 436 437]
       [438 439 440]
       [441 442 443]]
    
      [[444 445 446]
       [447 448 449]
       [450 451 452]
       [453 454 455]]
    
      [[456 457 458]
       [459 460 461]
       [462 463 464]
       [465 466 467]]
    
      [[468 469 470]
       [471 472 473]
       [474 475 476]
       [477 478 479]]]]
    (10, 4, 12)
    [[[  0   1   2   3   4   5   6   7   8   9  10  11]
      [ 12  13  14  15  16  17  18  19  20  21  22  23]
      [ 24  25  26  27  28  29  30  31  32  33  34  35]
      [ 36  37  38  39  40  41  42  43  44  45  46  47]]
    
     [[ 48  49  50  51  52  53  54  55  56  57  58  59]
      [ 60  61  62  63  64  65  66  67  68  69  70  71]
      [ 72  73  74  75  76  77  78  79  80  81  82  83]
      [ 84  85  86  87  88  89  90  91  92  93  94  95]]
    
     [[ 96  97  98  99 100 101 102 103 104 105 106 107]
      [108 109 110 111 112 113 114 115 116 117 118 119]
      [120 121 122 123 124 125 126 127 128 129 130 131]
      [132 133 134 135 136 137 138 139 140 141 142 143]]
    
     [[144 145 146 147 148 149 150 151 152 153 154 155]
      [156 157 158 159 160 161 162 163 164 165 166 167]
      [168 169 170 171 172 173 174 175 176 177 178 179]
      [180 181 182 183 184 185 186 187 188 189 190 191]]
    
     [[192 193 194 195 196 197 198 199 200 201 202 203]
      [204 205 206 207 208 209 210 211 212 213 214 215]
      [216 217 218 219 220 221 222 223 224 225 226 227]
      [228 229 230 231 232 233 234 235 236 237 238 239]]
    
     [[240 241 242 243 244 245 246 247 248 249 250 251]
      [252 253 254 255 256 257 258 259 260 261 262 263]
      [264 265 266 267 268 269 270 271 272 273 274 275]
      [276 277 278 279 280 281 282 283 284 285 286 287]]
    
     [[288 289 290 291 292 293 294 295 296 297 298 299]
      [300 301 302 303 304 305 306 307 308 309 310 311]
      [312 313 314 315 316 317 318 319 320 321 322 323]
      [324 325 326 327 328 329 330 331 332 333 334 335]]
    
     [[336 337 338 339 340 341 342 343 344 345 346 347]
      [348 349 350 351 352 353 354 355 356 357 358 359]
      [360 361 362 363 364 365 366 367 368 369 370 371]
      [372 373 374 375 376 377 378 379 380 381 382 383]]
    
     [[384 385 386 387 388 389 390 391 392 393 394 395]
      [396 397 398 399 400 401 402 403 404 405 406 407]
      [408 409 410 411 412 413 414 415 416 417 418 419]
      [420 421 422 423 424 425 426 427 428 429 430 431]]
    
     [[432 433 434 435 436 437 438 439 440 441 442 443]
      [444 445 446 447 448 449 450 451 452 453 454 455]
      [456 457 458 459 460 461 462 463 464 465 466 467]
      [468 469 470 471 472 473 474 475 476 477 478 479]]]
    (10, 48)
    [[  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17
       18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35
       36  37  38  39  40  41  42  43  44  45  46  47]
     [ 48  49  50  51  52  53  54  55  56  57  58  59  60  61  62  63  64  65
       66  67  68  69  70  71  72  73  74  75  76  77  78  79  80  81  82  83
       84  85  86  87  88  89  90  91  92  93  94  95]
     [ 96  97  98  99 100 101 102 103 104 105 106 107 108 109 110 111 112 113
      114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131
      132 133 134 135 136 137 138 139 140 141 142 143]
     [144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161
      162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179
      180 181 182 183 184 185 186 187 188 189 190 191]
     [192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209
      210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227
      228 229 230 231 232 233 234 235 236 237 238 239]
     [240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257
      258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274 275
      276 277 278 279 280 281 282 283 284 285 286 287]
     [288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305
      306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323
      324 325 326 327 328 329 330 331 332 333 334 335]
     [336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353
      354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371
      372 373 374 375 376 377 378 379 380 381 382 383]
     [384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401
      402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419
      420 421 422 423 424 425 426 427 428 429 430 431]
     [432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449
      450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467
      468 469 470 471 472 473 474 475 476 477 478 479]]
    (48, 10)
    [[  0  48  96 144 192 240 288 336 384 432]
     [  1  49  97 145 193 241 289 337 385 433]
     [  2  50  98 146 194 242 290 338 386 434]
     [  3  51  99 147 195 243 291 339 387 435]
     [  4  52 100 148 196 244 292 340 388 436]
     [  5  53 101 149 197 245 293 341 389 437]
     [  6  54 102 150 198 246 294 342 390 438]
     [  7  55 103 151 199 247 295 343 391 439]
     [  8  56 104 152 200 248 296 344 392 440]
     [  9  57 105 153 201 249 297 345 393 441]
     [ 10  58 106 154 202 250 298 346 394 442]
     [ 11  59 107 155 203 251 299 347 395 443]
     [ 12  60 108 156 204 252 300 348 396 444]
     [ 13  61 109 157 205 253 301 349 397 445]
     [ 14  62 110 158 206 254 302 350 398 446]
     [ 15  63 111 159 207 255 303 351 399 447]
     [ 16  64 112 160 208 256 304 352 400 448]
     [ 17  65 113 161 209 257 305 353 401 449]
     [ 18  66 114 162 210 258 306 354 402 450]
     [ 19  67 115 163 211 259 307 355 403 451]
     [ 20  68 116 164 212 260 308 356 404 452]
     [ 21  69 117 165 213 261 309 357 405 453]
     [ 22  70 118 166 214 262 310 358 406 454]
     [ 23  71 119 167 215 263 311 359 407 455]
     [ 24  72 120 168 216 264 312 360 408 456]
     [ 25  73 121 169 217 265 313 361 409 457]
     [ 26  74 122 170 218 266 314 362 410 458]
     [ 27  75 123 171 219 267 315 363 411 459]
     [ 28  76 124 172 220 268 316 364 412 460]
     [ 29  77 125 173 221 269 317 365 413 461]
     [ 30  78 126 174 222 270 318 366 414 462]
     [ 31  79 127 175 223 271 319 367 415 463]
     [ 32  80 128 176 224 272 320 368 416 464]
     [ 33  81 129 177 225 273 321 369 417 465]
     [ 34  82 130 178 226 274 322 370 418 466]
     [ 35  83 131 179 227 275 323 371 419 467]
     [ 36  84 132 180 228 276 324 372 420 468]
     [ 37  85 133 181 229 277 325 373 421 469]
     [ 38  86 134 182 230 278 326 374 422 470]
     [ 39  87 135 183 231 279 327 375 423 471]
     [ 40  88 136 184 232 280 328 376 424 472]
     [ 41  89 137 185 233 281 329 377 425 473]
     [ 42  90 138 186 234 282 330 378 426 474]
     [ 43  91 139 187 235 283 331 379 427 475]
     [ 44  92 140 188 236 284 332 380 428 476]
     [ 45  93 141 189 237 285 333 381 429 477]
     [ 46  94 142 190 238 286 334 382 430 478]
     [ 47  95 143 191 239 287 335 383 431 479]]
    
