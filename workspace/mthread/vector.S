
vector:     file format elf64-x86-64


Disassembly of section .init:

0000000000401000 <_init>:
  401000:	f3 0f 1e fa          	endbr64 
  401004:	48 83 ec 08          	sub    $0x8,%rsp
  401008:	48 8b 05 e9 5f 00 00 	mov    0x5fe9(%rip),%rax        # 406ff8 <__gmon_start__>
  40100f:	48 85 c0             	test   %rax,%rax
  401012:	74 02                	je     401016 <_init+0x16>
  401014:	ff d0                	callq  *%rax
  401016:	48 83 c4 08          	add    $0x8,%rsp
  40101a:	c3                   	retq   

Disassembly of section .plt:

0000000000401020 <.plt>:
  401020:	ff 35 e2 5f 00 00    	pushq  0x5fe2(%rip)        # 407008 <_GLOBAL_OFFSET_TABLE_+0x8>
  401026:	ff 25 e4 5f 00 00    	jmpq   *0x5fe4(%rip)        # 407010 <_GLOBAL_OFFSET_TABLE_+0x10>
  40102c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000401030 <printf@plt>:
  401030:	ff 25 e2 5f 00 00    	jmpq   *0x5fe2(%rip)        # 407018 <printf@GLIBC_2.2.5>
  401036:	68 00 00 00 00       	pushq  $0x0
  40103b:	e9 e0 ff ff ff       	jmpq   401020 <.plt>

0000000000401040 <cudaMalloc@plt>:
  401040:	ff 25 da 5f 00 00    	jmpq   *0x5fda(%rip)        # 407020 <cudaMalloc@libcudart.so.11.0>
  401046:	68 01 00 00 00       	pushq  $0x1
  40104b:	e9 d0 ff ff ff       	jmpq   401020 <.plt>

0000000000401050 <pthread_join@plt>:
  401050:	ff 25 d2 5f 00 00    	jmpq   *0x5fd2(%rip)        # 407028 <pthread_join@GLIBC_2.2.5>
  401056:	68 02 00 00 00       	pushq  $0x2
  40105b:	e9 c0 ff ff ff       	jmpq   401020 <.plt>

0000000000401060 <cudaLaunchKernel@plt>:
  401060:	ff 25 ca 5f 00 00    	jmpq   *0x5fca(%rip)        # 407030 <cudaLaunchKernel@libcudart.so.11.0>
  401066:	68 03 00 00 00       	pushq  $0x3
  40106b:	e9 b0 ff ff ff       	jmpq   401020 <.plt>

0000000000401070 <memset@plt>:
  401070:	ff 25 c2 5f 00 00    	jmpq   *0x5fc2(%rip)        # 407038 <memset@GLIBC_2.2.5>
  401076:	68 04 00 00 00       	pushq  $0x4
  40107b:	e9 a0 ff ff ff       	jmpq   401020 <.plt>

0000000000401080 <pthread_create@plt>:
  401080:	ff 25 ba 5f 00 00    	jmpq   *0x5fba(%rip)        # 407040 <pthread_create@GLIBC_2.2.5>
  401086:	68 05 00 00 00       	pushq  $0x5
  40108b:	e9 90 ff ff ff       	jmpq   401020 <.plt>

0000000000401090 <clock@plt>:
  401090:	ff 25 b2 5f 00 00    	jmpq   *0x5fb2(%rip)        # 407048 <clock@GLIBC_2.2.5>
  401096:	68 06 00 00 00       	pushq  $0x6
  40109b:	e9 80 ff ff ff       	jmpq   401020 <.plt>

00000000004010a0 <__cudaRegisterFatBinaryEnd@plt>:
  4010a0:	ff 25 aa 5f 00 00    	jmpq   *0x5faa(%rip)        # 407050 <__cudaRegisterFatBinaryEnd@libcudart.so.11.0>
  4010a6:	68 07 00 00 00       	pushq  $0x7
  4010ab:	e9 70 ff ff ff       	jmpq   401020 <.plt>

00000000004010b0 <cudaMemcpy@plt>:
  4010b0:	ff 25 a2 5f 00 00    	jmpq   *0x5fa2(%rip)        # 407058 <cudaMemcpy@libcudart.so.11.0>
  4010b6:	68 08 00 00 00       	pushq  $0x8
  4010bb:	e9 60 ff ff ff       	jmpq   401020 <.plt>

00000000004010c0 <__cudaRegisterFatBinary@plt>:
  4010c0:	ff 25 9a 5f 00 00    	jmpq   *0x5f9a(%rip)        # 407060 <__cudaRegisterFatBinary@libcudart.so.11.0>
  4010c6:	68 09 00 00 00       	pushq  $0x9
  4010cb:	e9 50 ff ff ff       	jmpq   401020 <.plt>

00000000004010d0 <__cxa_atexit@plt>:
  4010d0:	ff 25 92 5f 00 00    	jmpq   *0x5f92(%rip)        # 407068 <__cxa_atexit@GLIBC_2.2.5>
  4010d6:	68 0a 00 00 00       	pushq  $0xa
  4010db:	e9 40 ff ff ff       	jmpq   401020 <.plt>

00000000004010e0 <__cudaUnregisterFatBinary@plt>:
  4010e0:	ff 25 8a 5f 00 00    	jmpq   *0x5f8a(%rip)        # 407070 <__cudaUnregisterFatBinary@libcudart.so.11.0>
  4010e6:	68 0b 00 00 00       	pushq  $0xb
  4010eb:	e9 30 ff ff ff       	jmpq   401020 <.plt>

00000000004010f0 <__stack_chk_fail@plt>:
  4010f0:	ff 25 82 5f 00 00    	jmpq   *0x5f82(%rip)        # 407078 <__stack_chk_fail@GLIBC_2.4>
  4010f6:	68 0c 00 00 00       	pushq  $0xc
  4010fb:	e9 20 ff ff ff       	jmpq   401020 <.plt>

0000000000401100 <__cudaRegisterFunction@plt>:
  401100:	ff 25 7a 5f 00 00    	jmpq   *0x5f7a(%rip)        # 407080 <__cudaRegisterFunction@libcudart.so.11.0>
  401106:	68 0d 00 00 00       	pushq  $0xd
  40110b:	e9 10 ff ff ff       	jmpq   401020 <.plt>

0000000000401110 <free@plt>:
  401110:	ff 25 72 5f 00 00    	jmpq   *0x5f72(%rip)        # 407088 <free@GLIBC_2.2.5>
  401116:	68 0e 00 00 00       	pushq  $0xe
  40111b:	e9 00 ff ff ff       	jmpq   401020 <.plt>

0000000000401120 <__cudaInitModule@plt>:
  401120:	ff 25 6a 5f 00 00    	jmpq   *0x5f6a(%rip)        # 407090 <__cudaInitModule@libcudart.so.11.0>
  401126:	68 0f 00 00 00       	pushq  $0xf
  40112b:	e9 f0 fe ff ff       	jmpq   401020 <.plt>

0000000000401130 <malloc@plt>:
  401130:	ff 25 62 5f 00 00    	jmpq   *0x5f62(%rip)        # 407098 <malloc@GLIBC_2.2.5>
  401136:	68 10 00 00 00       	pushq  $0x10
  40113b:	e9 e0 fe ff ff       	jmpq   401020 <.plt>

0000000000401140 <__cudaPushCallConfiguration@plt>:
  401140:	ff 25 5a 5f 00 00    	jmpq   *0x5f5a(%rip)        # 4070a0 <__cudaPushCallConfiguration@libcudart.so.11.0>
  401146:	68 11 00 00 00       	pushq  $0x11
  40114b:	e9 d0 fe ff ff       	jmpq   401020 <.plt>

0000000000401150 <__cudaPopCallConfiguration@plt>:
  401150:	ff 25 52 5f 00 00    	jmpq   *0x5f52(%rip)        # 4070a8 <__cudaPopCallConfiguration@libcudart.so.11.0>
  401156:	68 12 00 00 00       	pushq  $0x12
  40115b:	e9 c0 fe ff ff       	jmpq   401020 <.plt>

0000000000401160 <__gxx_personality_v0@plt>:
  401160:	ff 25 4a 5f 00 00    	jmpq   *0x5f4a(%rip)        # 4070b0 <__gxx_personality_v0@CXXABI_1.3>
  401166:	68 13 00 00 00       	pushq  $0x13
  40116b:	e9 b0 fe ff ff       	jmpq   401020 <.plt>

0000000000401170 <_Unwind_Resume@plt>:
  401170:	ff 25 42 5f 00 00    	jmpq   *0x5f42(%rip)        # 4070b8 <_Unwind_Resume@GCC_3.0>
  401176:	68 14 00 00 00       	pushq  $0x14
  40117b:	e9 a0 fe ff ff       	jmpq   401020 <.plt>

0000000000401180 <pthread_exit@plt>:
  401180:	ff 25 3a 5f 00 00    	jmpq   *0x5f3a(%rip)        # 4070c0 <pthread_exit@GLIBC_2.2.5>
  401186:	68 15 00 00 00       	pushq  $0x15
  40118b:	e9 90 fe ff ff       	jmpq   401020 <.plt>

0000000000401190 <cudaFree@plt>:
  401190:	ff 25 32 5f 00 00    	jmpq   *0x5f32(%rip)        # 4070c8 <cudaFree@libcudart.so.11.0>
  401196:	68 16 00 00 00       	pushq  $0x16
  40119b:	e9 80 fe ff ff       	jmpq   401020 <.plt>

Disassembly of section .text:

00000000004011a0 <_start>:
  4011a0:	f3 0f 1e fa          	endbr64 
  4011a4:	31 ed                	xor    %ebp,%ebp
  4011a6:	49 89 d1             	mov    %rdx,%r9
  4011a9:	5e                   	pop    %rsi
  4011aa:	48 89 e2             	mov    %rsp,%rdx
  4011ad:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  4011b1:	50                   	push   %rax
  4011b2:	54                   	push   %rsp
  4011b3:	49 c7 c0 60 25 40 00 	mov    $0x402560,%r8
  4011ba:	48 c7 c1 f0 24 40 00 	mov    $0x4024f0,%rcx
  4011c1:	48 c7 c7 4b 17 40 00 	mov    $0x40174b,%rdi
  4011c8:	ff 15 22 5e 00 00    	callq  *0x5e22(%rip)        # 406ff0 <__libc_start_main@GLIBC_2.2.5>
  4011ce:	f4                   	hlt    
  4011cf:	90                   	nop

00000000004011d0 <_dl_relocate_static_pie>:
  4011d0:	f3 0f 1e fa          	endbr64 
  4011d4:	c3                   	retq   
  4011d5:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4011dc:	00 00 00 
  4011df:	90                   	nop

00000000004011e0 <deregister_tm_clones>:
  4011e0:	b8 e7 70 40 00       	mov    $0x4070e7,%eax
  4011e5:	55                   	push   %rbp
  4011e6:	48 2d e0 70 40 00    	sub    $0x4070e0,%rax
  4011ec:	48 83 f8 0e          	cmp    $0xe,%rax
  4011f0:	48 89 e5             	mov    %rsp,%rbp
  4011f3:	76 1b                	jbe    401210 <deregister_tm_clones+0x30>
  4011f5:	b8 00 00 00 00       	mov    $0x0,%eax
  4011fa:	48 85 c0             	test   %rax,%rax
  4011fd:	74 11                	je     401210 <deregister_tm_clones+0x30>
  4011ff:	5d                   	pop    %rbp
  401200:	bf e0 70 40 00       	mov    $0x4070e0,%edi
  401205:	ff e0                	jmpq   *%rax
  401207:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  40120e:	00 00 
  401210:	5d                   	pop    %rbp
  401211:	c3                   	retq   
  401212:	0f 1f 40 00          	nopl   0x0(%rax)
  401216:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40121d:	00 00 00 

0000000000401220 <register_tm_clones>:
  401220:	be e0 70 40 00       	mov    $0x4070e0,%esi
  401225:	55                   	push   %rbp
  401226:	48 81 ee e0 70 40 00 	sub    $0x4070e0,%rsi
  40122d:	48 c1 fe 03          	sar    $0x3,%rsi
  401231:	48 89 e5             	mov    %rsp,%rbp
  401234:	48 89 f0             	mov    %rsi,%rax
  401237:	48 c1 e8 3f          	shr    $0x3f,%rax
  40123b:	48 01 c6             	add    %rax,%rsi
  40123e:	48 d1 fe             	sar    %rsi
  401241:	74 15                	je     401258 <register_tm_clones+0x38>
  401243:	b8 00 00 00 00       	mov    $0x0,%eax
  401248:	48 85 c0             	test   %rax,%rax
  40124b:	74 0b                	je     401258 <register_tm_clones+0x38>
  40124d:	5d                   	pop    %rbp
  40124e:	bf e0 70 40 00       	mov    $0x4070e0,%edi
  401253:	ff e0                	jmpq   *%rax
  401255:	0f 1f 00             	nopl   (%rax)
  401258:	5d                   	pop    %rbp
  401259:	c3                   	retq   
  40125a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000401260 <__do_global_dtors_aux>:
  401260:	80 3d 79 5e 00 00 00 	cmpb   $0x0,0x5e79(%rip)        # 4070e0 <__TMC_END__>
  401267:	75 11                	jne    40127a <__do_global_dtors_aux+0x1a>
  401269:	55                   	push   %rbp
  40126a:	48 89 e5             	mov    %rsp,%rbp
  40126d:	e8 6e ff ff ff       	callq  4011e0 <deregister_tm_clones>
  401272:	5d                   	pop    %rbp
  401273:	c6 05 66 5e 00 00 01 	movb   $0x1,0x5e66(%rip)        # 4070e0 <__TMC_END__>
  40127a:	f3 c3                	repz retq 
  40127c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000401280 <frame_dummy>:
  401280:	bf d8 6d 40 00       	mov    $0x406dd8,%edi
  401285:	48 83 3f 00          	cmpq   $0x0,(%rdi)
  401289:	75 05                	jne    401290 <frame_dummy+0x10>
  40128b:	eb 93                	jmp    401220 <register_tm_clones>
  40128d:	0f 1f 00             	nopl   (%rax)
  401290:	b8 00 00 00 00       	mov    $0x0,%eax
  401295:	48 85 c0             	test   %rax,%rax
  401298:	74 f1                	je     40128b <frame_dummy+0xb>
  40129a:	55                   	push   %rbp
  40129b:	48 89 e5             	mov    %rsp,%rbp
  40129e:	ff d0                	callq  *%rax
  4012a0:	5d                   	pop    %rbp
  4012a1:	e9 7a ff ff ff       	jmpq   401220 <register_tm_clones>

00000000004012a6 <__cudaUnregisterBinaryUtil>:
  4012a6:	55                   	push   %rbp
  4012a7:	48 89 e5             	mov    %rsp,%rbp
  4012aa:	48 8b 05 37 5e 00 00 	mov    0x5e37(%rip),%rax        # 4070e8 <_ZL20__cudaFatCubinHandle>
  4012b1:	48 89 c7             	mov    %rax,%rdi
  4012b4:	e8 27 fe ff ff       	callq  4010e0 <__cudaUnregisterFatBinary@plt>
  4012b9:	90                   	nop
  4012ba:	5d                   	pop    %rbp
  4012bb:	c3                   	retq   

00000000004012bc <_Z7thread0Pv>:
  4012bc:	55                   	push   %rbp
  4012bd:	48 89 e5             	mov    %rsp,%rbp
  4012c0:	48 83 ec 40          	sub    $0x40,%rsp
  4012c4:	48 89 7d c8          	mov    %rdi,-0x38(%rbp)
  4012c8:	e8 c3 fd ff ff       	callq  401090 <clock@plt>
  4012cd:	66 0f ef c0          	pxor   %xmm0,%xmm0
  4012d1:	f3 48 0f 2a c0       	cvtsi2ss %rax,%xmm0
  4012d6:	f3 0f 11 45 dc       	movss  %xmm0,-0x24(%rbp)
  4012db:	bf 08 30 40 00       	mov    $0x403008,%edi
  4012e0:	b8 00 00 00 00       	mov    $0x0,%eax
  4012e5:	e8 46 fd ff ff       	callq  401030 <printf@plt>
  4012ea:	c7 45 e0 64 00 00 00 	movl   $0x64,-0x20(%rbp)
  4012f1:	8b 45 e0             	mov    -0x20(%rbp),%eax
  4012f4:	48 98                	cltq   
  4012f6:	48 c1 e0 02          	shl    $0x2,%rax
  4012fa:	48 89 c7             	mov    %rax,%rdi
  4012fd:	e8 2e fe ff ff       	callq  401130 <malloc@plt>
  401302:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  401306:	8b 45 e0             	mov    -0x20(%rbp),%eax
  401309:	48 98                	cltq   
  40130b:	48 c1 e0 02          	shl    $0x2,%rax
  40130f:	48 89 c7             	mov    %rax,%rdi
  401312:	e8 19 fe ff ff       	callq  401130 <malloc@plt>
  401317:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  40131b:	8b 45 e0             	mov    -0x20(%rbp),%eax
  40131e:	48 98                	cltq   
  401320:	48 c1 e0 02          	shl    $0x2,%rax
  401324:	48 89 c7             	mov    %rax,%rdi
  401327:	e8 04 fe ff ff       	callq  401130 <malloc@plt>
  40132c:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  401330:	c7 45 d8 00 00 00 00 	movl   $0x0,-0x28(%rbp)
  401337:	8b 45 d8             	mov    -0x28(%rbp),%eax
  40133a:	3b 45 e0             	cmp    -0x20(%rbp),%eax
  40133d:	7d 46                	jge    401385 <_Z7thread0Pv+0xc9>
  40133f:	8b 45 d8             	mov    -0x28(%rbp),%eax
  401342:	48 98                	cltq   
  401344:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  40134b:	00 
  40134c:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  401350:	48 01 d0             	add    %rdx,%rax
  401353:	f3 0f 10 05 3d 1e 00 	movss  0x1e3d(%rip),%xmm0        # 403198 <_IO_stdin_used+0x198>
  40135a:	00 
  40135b:	f3 0f 11 00          	movss  %xmm0,(%rax)
  40135f:	8b 45 d8             	mov    -0x28(%rbp),%eax
  401362:	48 98                	cltq   
  401364:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  40136b:	00 
  40136c:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  401370:	48 01 d0             	add    %rdx,%rax
  401373:	f3 0f 10 05 1d 1e 00 	movss  0x1e1d(%rip),%xmm0        # 403198 <_IO_stdin_used+0x198>
  40137a:	00 
  40137b:	f3 0f 11 00          	movss  %xmm0,(%rax)
  40137f:	83 45 d8 01          	addl   $0x1,-0x28(%rbp)
  401383:	eb b2                	jmp    401337 <_Z7thread0Pv+0x7b>
  401385:	c7 45 d8 00 00 00 00 	movl   $0x0,-0x28(%rbp)
  40138c:	83 7d d8 09          	cmpl   $0x9,-0x28(%rbp)
  401390:	7f 1d                	jg     4013af <_Z7thread0Pv+0xf3>
  401392:	8b 4d e0             	mov    -0x20(%rbp),%ecx
  401395:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
  401399:	48 8b 75 f0          	mov    -0x10(%rbp),%rsi
  40139d:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4013a1:	48 89 c7             	mov    %rax,%rdi
  4013a4:	e8 3e 04 00 00       	callq  4017e7 <_Z9vectorAddPfS_S_i>
  4013a9:	83 45 d8 01          	addl   $0x1,-0x28(%rbp)
  4013ad:	eb dd                	jmp    40138c <_Z7thread0Pv+0xd0>
  4013af:	c7 45 d8 00 00 00 00 	movl   $0x0,-0x28(%rbp)
  4013b6:	8b 45 d8             	mov    -0x28(%rbp),%eax
  4013b9:	3b 45 e0             	cmp    -0x20(%rbp),%eax
  4013bc:	7d 36                	jge    4013f4 <_Z7thread0Pv+0x138>
  4013be:	8b 45 d8             	mov    -0x28(%rbp),%eax
  4013c1:	48 98                	cltq   
  4013c3:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  4013ca:	00 
  4013cb:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4013cf:	48 01 d0             	add    %rdx,%rax
  4013d2:	f3 0f 10 00          	movss  (%rax),%xmm0
  4013d6:	f3 0f 5a c0          	cvtss2sd %xmm0,%xmm0
  4013da:	8b 45 d8             	mov    -0x28(%rbp),%eax
  4013dd:	89 c6                	mov    %eax,%esi
  4013df:	bf 1b 30 40 00       	mov    $0x40301b,%edi
  4013e4:	b8 01 00 00 00       	mov    $0x1,%eax
  4013e9:	e8 42 fc ff ff       	callq  401030 <printf@plt>
  4013ee:	83 45 d8 01          	addl   $0x1,-0x28(%rbp)
  4013f2:	eb c2                	jmp    4013b6 <_Z7thread0Pv+0xfa>
  4013f4:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4013f8:	48 89 c7             	mov    %rax,%rdi
  4013fb:	e8 10 fd ff ff       	callq  401110 <free@plt>
  401400:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  401404:	48 89 c7             	mov    %rax,%rdi
  401407:	e8 04 fd ff ff       	callq  401110 <free@plt>
  40140c:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  401410:	48 89 c7             	mov    %rax,%rdi
  401413:	e8 f8 fc ff ff       	callq  401110 <free@plt>
  401418:	e8 73 fc ff ff       	callq  401090 <clock@plt>
  40141d:	66 0f ef c0          	pxor   %xmm0,%xmm0
  401421:	f3 48 0f 2a c0       	cvtsi2ss %rax,%xmm0
  401426:	f3 0f 5c 45 dc       	subss  -0x24(%rbp),%xmm0
  40142b:	f3 0f 11 45 e4       	movss  %xmm0,-0x1c(%rbp)
  401430:	f3 0f 10 45 e4       	movss  -0x1c(%rbp),%xmm0
  401435:	f3 0f 10 0d 5f 1d 00 	movss  0x1d5f(%rip),%xmm1        # 40319c <_IO_stdin_used+0x19c>
  40143c:	00 
  40143d:	f3 0f 5e c1          	divss  %xmm1,%xmm0
  401441:	f3 0f 5a c0          	cvtss2sd %xmm0,%xmm0
  401445:	bf 38 30 40 00       	mov    $0x403038,%edi
  40144a:	b8 01 00 00 00       	mov    $0x1,%eax
  40144f:	e8 dc fb ff ff       	callq  401030 <printf@plt>
  401454:	bf 00 00 00 00       	mov    $0x0,%edi
  401459:	e8 22 fd ff ff       	callq  401180 <pthread_exit@plt>

000000000040145e <_Z7thread1Pv>:
  40145e:	55                   	push   %rbp
  40145f:	48 89 e5             	mov    %rsp,%rbp
  401462:	48 83 ec 40          	sub    $0x40,%rsp
  401466:	48 89 7d c8          	mov    %rdi,-0x38(%rbp)
  40146a:	e8 21 fc ff ff       	callq  401090 <clock@plt>
  40146f:	66 0f ef c0          	pxor   %xmm0,%xmm0
  401473:	f3 48 0f 2a c0       	cvtsi2ss %rax,%xmm0
  401478:	f3 0f 11 45 d4       	movss  %xmm0,-0x2c(%rbp)
  40147d:	bf 68 30 40 00       	mov    $0x403068,%edi
  401482:	b8 00 00 00 00       	mov    $0x0,%eax
  401487:	e8 a4 fb ff ff       	callq  401030 <printf@plt>
  40148c:	c7 45 d8 64 00 00 00 	movl   $0x64,-0x28(%rbp)
  401493:	8b 45 d8             	mov    -0x28(%rbp),%eax
  401496:	48 98                	cltq   
  401498:	48 c1 e0 02          	shl    $0x2,%rax
  40149c:	48 89 c7             	mov    %rax,%rdi
  40149f:	e8 8c fc ff ff       	callq  401130 <malloc@plt>
  4014a4:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
  4014a8:	8b 45 d8             	mov    -0x28(%rbp),%eax
  4014ab:	48 98                	cltq   
  4014ad:	48 c1 e0 02          	shl    $0x2,%rax
  4014b1:	48 89 c7             	mov    %rax,%rdi
  4014b4:	e8 77 fc ff ff       	callq  401130 <malloc@plt>
  4014b9:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
  4014bd:	8b 45 d8             	mov    -0x28(%rbp),%eax
  4014c0:	48 98                	cltq   
  4014c2:	48 c1 e0 02          	shl    $0x2,%rax
  4014c6:	48 89 c7             	mov    %rax,%rdi
  4014c9:	e8 62 fc ff ff       	callq  401130 <malloc@plt>
  4014ce:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
  4014d2:	8b 45 d8             	mov    -0x28(%rbp),%eax
  4014d5:	48 98                	cltq   
  4014d7:	48 c1 e0 02          	shl    $0x2,%rax
  4014db:	48 89 c7             	mov    %rax,%rdi
  4014de:	e8 4d fc ff ff       	callq  401130 <malloc@plt>
  4014e3:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  4014e7:	c7 45 d0 00 00 00 00 	movl   $0x0,-0x30(%rbp)
  4014ee:	8b 45 d0             	mov    -0x30(%rbp),%eax
  4014f1:	3b 45 d8             	cmp    -0x28(%rbp),%eax
  4014f4:	7d 53                	jge    401549 <_Z7thread1Pv+0xeb>
  4014f6:	8b 45 d0             	mov    -0x30(%rbp),%eax
  4014f9:	48 98                	cltq   
  4014fb:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  401502:	00 
  401503:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  401507:	48 01 d0             	add    %rdx,%rax
  40150a:	66 0f ef c0          	pxor   %xmm0,%xmm0
  40150e:	f3 0f 2a 45 d0       	cvtsi2ssl -0x30(%rbp),%xmm0
  401513:	f3 0f 10 0d 85 1c 00 	movss  0x1c85(%rip),%xmm1        # 4031a0 <_IO_stdin_used+0x1a0>
  40151a:	00 
  40151b:	f3 0f 58 c1          	addss  %xmm1,%xmm0
  40151f:	f3 0f 11 00          	movss  %xmm0,(%rax)
  401523:	8b 45 d0             	mov    -0x30(%rbp),%eax
  401526:	48 98                	cltq   
  401528:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  40152f:	00 
  401530:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  401534:	48 01 d0             	add    %rdx,%rax
  401537:	f3 0f 10 05 65 1c 00 	movss  0x1c65(%rip),%xmm0        # 4031a4 <_IO_stdin_used+0x1a4>
  40153e:	00 
  40153f:	f3 0f 11 00          	movss  %xmm0,(%rax)
  401543:	83 45 d0 01          	addl   $0x1,-0x30(%rbp)
  401547:	eb a5                	jmp    4014ee <_Z7thread1Pv+0x90>
  401549:	c7 45 d0 00 00 00 00 	movl   $0x0,-0x30(%rbp)
  401550:	83 7d d0 09          	cmpl   $0x9,-0x30(%rbp)
  401554:	7f 34                	jg     40158a <_Z7thread1Pv+0x12c>
  401556:	8b 4d d8             	mov    -0x28(%rbp),%ecx
  401559:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
  40155d:	48 8b 75 e8          	mov    -0x18(%rbp),%rsi
  401561:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  401565:	48 89 c7             	mov    %rax,%rdi
  401568:	e8 34 04 00 00       	callq  4019a1 <_Z9vectorSubPfS_S_i>
  40156d:	8b 4d d8             	mov    -0x28(%rbp),%ecx
  401570:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
  401574:	48 8b 75 e8          	mov    -0x18(%rbp),%rsi
  401578:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  40157c:	48 89 c7             	mov    %rax,%rdi
  40157f:	e8 d7 05 00 00       	callq  401b5b <_Z9vectorMulPfS_S_i>
  401584:	83 45 d0 01          	addl   $0x1,-0x30(%rbp)
  401588:	eb c6                	jmp    401550 <_Z7thread1Pv+0xf2>
  40158a:	c7 45 d0 00 00 00 00 	movl   $0x0,-0x30(%rbp)
  401591:	8b 45 d0             	mov    -0x30(%rbp),%eax
  401594:	3b 45 d8             	cmp    -0x28(%rbp),%eax
  401597:	7d 36                	jge    4015cf <_Z7thread1Pv+0x171>
  401599:	8b 45 d0             	mov    -0x30(%rbp),%eax
  40159c:	48 98                	cltq   
  40159e:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  4015a5:	00 
  4015a6:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4015aa:	48 01 d0             	add    %rdx,%rax
  4015ad:	f3 0f 10 00          	movss  (%rax),%xmm0
  4015b1:	f3 0f 5a c0          	cvtss2sd %xmm0,%xmm0
  4015b5:	8b 45 d0             	mov    -0x30(%rbp),%eax
  4015b8:	89 c6                	mov    %eax,%esi
  4015ba:	bf 7a 30 40 00       	mov    $0x40307a,%edi
  4015bf:	b8 01 00 00 00       	mov    $0x1,%eax
  4015c4:	e8 67 fa ff ff       	callq  401030 <printf@plt>
  4015c9:	83 45 d0 01          	addl   $0x1,-0x30(%rbp)
  4015cd:	eb c2                	jmp    401591 <_Z7thread1Pv+0x133>
  4015cf:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  4015d3:	48 89 c7             	mov    %rax,%rdi
  4015d6:	e8 35 fb ff ff       	callq  401110 <free@plt>
  4015db:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
  4015df:	48 89 c7             	mov    %rax,%rdi
  4015e2:	e8 29 fb ff ff       	callq  401110 <free@plt>
  4015e7:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
  4015eb:	48 89 c7             	mov    %rax,%rdi
  4015ee:	e8 1d fb ff ff       	callq  401110 <free@plt>
  4015f3:	e8 98 fa ff ff       	callq  401090 <clock@plt>
  4015f8:	66 0f ef c0          	pxor   %xmm0,%xmm0
  4015fc:	f3 48 0f 2a c0       	cvtsi2ss %rax,%xmm0
  401601:	f3 0f 5c 45 d4       	subss  -0x2c(%rbp),%xmm0
  401606:	f3 0f 11 45 dc       	movss  %xmm0,-0x24(%rbp)
  40160b:	f3 0f 10 45 dc       	movss  -0x24(%rbp),%xmm0
  401610:	f3 0f 10 0d 84 1b 00 	movss  0x1b84(%rip),%xmm1        # 40319c <_IO_stdin_used+0x19c>
  401617:	00 
  401618:	f3 0f 5e c1          	divss  %xmm1,%xmm0
  40161c:	f3 0f 5a c0          	cvtss2sd %xmm0,%xmm0
  401620:	bf 98 30 40 00       	mov    $0x403098,%edi
  401625:	b8 01 00 00 00       	mov    $0x1,%eax
  40162a:	e8 01 fa ff ff       	callq  401030 <printf@plt>
  40162f:	bf 00 00 00 00       	mov    $0x0,%edi
  401634:	e8 47 fb ff ff       	callq  401180 <pthread_exit@plt>

0000000000401639 <_Z13thread_createv>:
  401639:	55                   	push   %rbp
  40163a:	48 89 e5             	mov    %rsp,%rbp
  40163d:	48 83 ec 10          	sub    $0x10,%rsp
  401641:	ba 20 00 00 00       	mov    $0x20,%edx
  401646:	be 00 00 00 00       	mov    $0x0,%esi
  40164b:	bf 00 71 40 00       	mov    $0x407100,%edi
  401650:	e8 1b fa ff ff       	callq  401070 <memset@plt>
  401655:	b9 00 00 00 00       	mov    $0x0,%ecx
  40165a:	ba bc 12 40 00       	mov    $0x4012bc,%edx
  40165f:	be 00 00 00 00       	mov    $0x0,%esi
  401664:	bf 00 71 40 00       	mov    $0x407100,%edi
  401669:	e8 12 fa ff ff       	callq  401080 <pthread_create@plt>
  40166e:	89 45 fc             	mov    %eax,-0x4(%rbp)
  401671:	83 7d fc 00          	cmpl   $0x0,-0x4(%rbp)
  401675:	0f 95 c0             	setne  %al
  401678:	84 c0                	test   %al,%al
  40167a:	74 11                	je     40168d <_Z13thread_createv+0x54>
  40167c:	bf c8 30 40 00       	mov    $0x4030c8,%edi
  401681:	b8 00 00 00 00       	mov    $0x0,%eax
  401686:	e8 a5 f9 ff ff       	callq  401030 <printf@plt>
  40168b:	eb 0f                	jmp    40169c <_Z13thread_createv+0x63>
  40168d:	bf e4 30 40 00       	mov    $0x4030e4,%edi
  401692:	b8 00 00 00 00       	mov    $0x0,%eax
  401697:	e8 94 f9 ff ff       	callq  401030 <printf@plt>
  40169c:	b9 00 00 00 00       	mov    $0x0,%ecx
  4016a1:	ba 5e 14 40 00       	mov    $0x40145e,%edx
  4016a6:	be 00 00 00 00       	mov    $0x0,%esi
  4016ab:	bf 08 71 40 00       	mov    $0x407108,%edi
  4016b0:	e8 cb f9 ff ff       	callq  401080 <pthread_create@plt>
  4016b5:	89 45 fc             	mov    %eax,-0x4(%rbp)
  4016b8:	83 7d fc 00          	cmpl   $0x0,-0x4(%rbp)
  4016bc:	0f 95 c0             	setne  %al
  4016bf:	84 c0                	test   %al,%al
  4016c1:	74 11                	je     4016d4 <_Z13thread_createv+0x9b>
  4016c3:	bf f9 30 40 00       	mov    $0x4030f9,%edi
  4016c8:	b8 00 00 00 00       	mov    $0x0,%eax
  4016cd:	e8 5e f9 ff ff       	callq  401030 <printf@plt>
  4016d2:	eb 0f                	jmp    4016e3 <_Z13thread_createv+0xaa>
  4016d4:	bf 15 31 40 00       	mov    $0x403115,%edi
  4016d9:	b8 00 00 00 00       	mov    $0x0,%eax
  4016de:	e8 4d f9 ff ff       	callq  401030 <printf@plt>
  4016e3:	90                   	nop
  4016e4:	c9                   	leaveq 
  4016e5:	c3                   	retq   

00000000004016e6 <_Z11thread_waitv>:
  4016e6:	55                   	push   %rbp
  4016e7:	48 89 e5             	mov    %rsp,%rbp
  4016ea:	48 8b 05 0f 5a 00 00 	mov    0x5a0f(%rip),%rax        # 407100 <thread>
  4016f1:	48 85 c0             	test   %rax,%rax
  4016f4:	74 23                	je     401719 <_Z11thread_waitv+0x33>
  4016f6:	48 8b 05 03 5a 00 00 	mov    0x5a03(%rip),%rax        # 407100 <thread>
  4016fd:	be 00 00 00 00       	mov    $0x0,%esi
  401702:	48 89 c7             	mov    %rax,%rdi
  401705:	e8 46 f9 ff ff       	callq  401050 <pthread_join@plt>
  40170a:	bf 2a 31 40 00       	mov    $0x40312a,%edi
  40170f:	b8 00 00 00 00       	mov    $0x0,%eax
  401714:	e8 17 f9 ff ff       	callq  401030 <printf@plt>
  401719:	48 8b 05 e8 59 00 00 	mov    0x59e8(%rip),%rax        # 407108 <thread+0x8>
  401720:	48 85 c0             	test   %rax,%rax
  401723:	74 23                	je     401748 <_Z11thread_waitv+0x62>
  401725:	48 8b 05 dc 59 00 00 	mov    0x59dc(%rip),%rax        # 407108 <thread+0x8>
  40172c:	be 00 00 00 00       	mov    $0x0,%esi
  401731:	48 89 c7             	mov    %rax,%rdi
  401734:	e8 17 f9 ff ff       	callq  401050 <pthread_join@plt>
  401739:	bf 3b 31 40 00       	mov    $0x40313b,%edi
  40173e:	b8 00 00 00 00       	mov    $0x0,%eax
  401743:	e8 e8 f8 ff ff       	callq  401030 <printf@plt>
  401748:	90                   	nop
  401749:	5d                   	pop    %rbp
  40174a:	c3                   	retq   

000000000040174b <main>:
  40174b:	55                   	push   %rbp
  40174c:	48 89 e5             	mov    %rsp,%rbp
  40174f:	48 83 ec 10          	sub    $0x10,%rsp
  401753:	bf 4c 31 40 00       	mov    $0x40314c,%edi
  401758:	b8 00 00 00 00       	mov    $0x0,%eax
  40175d:	e8 ce f8 ff ff       	callq  401030 <printf@plt>
  401762:	e8 29 f9 ff ff       	callq  401090 <clock@plt>
  401767:	66 0f ef c0          	pxor   %xmm0,%xmm0
  40176b:	f3 48 0f 2a c0       	cvtsi2ss %rax,%xmm0
  401770:	f3 0f 11 45 f8       	movss  %xmm0,-0x8(%rbp)
  401775:	e8 bf fe ff ff       	callq  401639 <_Z13thread_createv>
  40177a:	bf 60 31 40 00       	mov    $0x403160,%edi
  40177f:	b8 00 00 00 00       	mov    $0x0,%eax
  401784:	e8 a7 f8 ff ff       	callq  401030 <printf@plt>
  401789:	e8 58 ff ff ff       	callq  4016e6 <_Z11thread_waitv>
  40178e:	e8 fd f8 ff ff       	callq  401090 <clock@plt>
  401793:	66 0f ef c0          	pxor   %xmm0,%xmm0
  401797:	f3 48 0f 2a c0       	cvtsi2ss %rax,%xmm0
  40179c:	f3 0f 5c 45 f8       	subss  -0x8(%rbp),%xmm0
  4017a1:	f3 0f 11 45 fc       	movss  %xmm0,-0x4(%rbp)
  4017a6:	f3 0f 10 45 fc       	movss  -0x4(%rbp),%xmm0
  4017ab:	f3 0f 10 0d e9 19 00 	movss  0x19e9(%rip),%xmm1        # 40319c <_IO_stdin_used+0x19c>
  4017b2:	00 
  4017b3:	f3 0f 5e c1          	divss  %xmm1,%xmm0
  4017b7:	f3 0f 5a c0          	cvtss2sd %xmm0,%xmm0
  4017bb:	bf 7b 31 40 00       	mov    $0x40317b,%edi
  4017c0:	b8 01 00 00 00       	mov    $0x1,%eax
  4017c5:	e8 66 f8 ff ff       	callq  401030 <printf@plt>
  4017ca:	b8 00 00 00 00       	mov    $0x0,%eax
  4017cf:	c9                   	leaveq 
  4017d0:	c3                   	retq   

00000000004017d1 <_ZL37__nv_save_fatbinhandle_for_managed_rtPPv>:
  4017d1:	55                   	push   %rbp
  4017d2:	48 89 e5             	mov    %rsp,%rbp
  4017d5:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4017d9:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4017dd:	48 89 05 44 59 00 00 	mov    %rax,0x5944(%rip)        # 407128 <_ZL32__nv_fatbinhandle_for_managed_rt>
  4017e4:	90                   	nop
  4017e5:	5d                   	pop    %rbp
  4017e6:	c3                   	retq   

00000000004017e7 <_Z9vectorAddPfS_S_i>:
  4017e7:	55                   	push   %rbp
  4017e8:	48 89 e5             	mov    %rsp,%rbp
  4017eb:	48 83 ec 70          	sub    $0x70,%rsp
  4017ef:	48 89 7d a8          	mov    %rdi,-0x58(%rbp)
  4017f3:	48 89 75 a0          	mov    %rsi,-0x60(%rbp)
  4017f7:	48 89 55 98          	mov    %rdx,-0x68(%rbp)
  4017fb:	89 4d 94             	mov    %ecx,-0x6c(%rbp)
  4017fe:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  401805:	00 00 
  401807:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  40180b:	31 c0                	xor    %eax,%eax
  40180d:	8b 45 94             	mov    -0x6c(%rbp),%eax
  401810:	48 98                	cltq   
  401812:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  401819:	00 
  40181a:	48 8d 45 b8          	lea    -0x48(%rbp),%rax
  40181e:	48 89 d6             	mov    %rdx,%rsi
  401821:	48 89 c7             	mov    %rax,%rdi
  401824:	e8 17 f8 ff ff       	callq  401040 <cudaMalloc@plt>
  401829:	8b 45 94             	mov    -0x6c(%rbp),%eax
  40182c:	48 98                	cltq   
  40182e:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  401835:	00 
  401836:	48 8b 45 b8          	mov    -0x48(%rbp),%rax
  40183a:	48 8b 75 a8          	mov    -0x58(%rbp),%rsi
  40183e:	b9 01 00 00 00       	mov    $0x1,%ecx
  401843:	48 89 c7             	mov    %rax,%rdi
  401846:	e8 65 f8 ff ff       	callq  4010b0 <cudaMemcpy@plt>
  40184b:	8b 45 94             	mov    -0x6c(%rbp),%eax
  40184e:	48 98                	cltq   
  401850:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  401857:	00 
  401858:	48 8d 45 c0          	lea    -0x40(%rbp),%rax
  40185c:	48 89 d6             	mov    %rdx,%rsi
  40185f:	48 89 c7             	mov    %rax,%rdi
  401862:	e8 d9 f7 ff ff       	callq  401040 <cudaMalloc@plt>
  401867:	8b 45 94             	mov    -0x6c(%rbp),%eax
  40186a:	48 98                	cltq   
  40186c:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  401873:	00 
  401874:	48 8b 45 c0          	mov    -0x40(%rbp),%rax
  401878:	48 8b 75 a0          	mov    -0x60(%rbp),%rsi
  40187c:	b9 01 00 00 00       	mov    $0x1,%ecx
  401881:	48 89 c7             	mov    %rax,%rdi
  401884:	e8 27 f8 ff ff       	callq  4010b0 <cudaMemcpy@plt>
  401889:	8b 45 94             	mov    -0x6c(%rbp),%eax
  40188c:	48 98                	cltq   
  40188e:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  401895:	00 
  401896:	48 8d 45 c8          	lea    -0x38(%rbp),%rax
  40189a:	48 89 d6             	mov    %rdx,%rsi
  40189d:	48 89 c7             	mov    %rax,%rdi
  4018a0:	e8 9b f7 ff ff       	callq  401040 <cudaMalloc@plt>
  4018a5:	8b 45 94             	mov    -0x6c(%rbp),%eax
  4018a8:	48 98                	cltq   
  4018aa:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  4018b1:	00 
  4018b2:	48 8b 45 c0          	mov    -0x40(%rbp),%rax
  4018b6:	48 8b 75 a0          	mov    -0x60(%rbp),%rsi
  4018ba:	b9 01 00 00 00       	mov    $0x1,%ecx
  4018bf:	48 89 c7             	mov    %rax,%rdi
  4018c2:	e8 e9 f7 ff ff       	callq  4010b0 <cudaMemcpy@plt>
  4018c7:	8b 75 94             	mov    -0x6c(%rbp),%esi
  4018ca:	48 8d 45 e0          	lea    -0x20(%rbp),%rax
  4018ce:	b9 01 00 00 00       	mov    $0x1,%ecx
  4018d3:	ba 01 00 00 00       	mov    $0x1,%edx
  4018d8:	48 89 c7             	mov    %rax,%rdi
  4018db:	e8 f8 04 00 00       	callq  401dd8 <_ZN4dim3C1Ejjj>
  4018e0:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  4018e4:	b9 01 00 00 00       	mov    $0x1,%ecx
  4018e9:	ba 01 00 00 00       	mov    $0x1,%edx
  4018ee:	be 01 00 00 00       	mov    $0x1,%esi
  4018f3:	48 89 c7             	mov    %rax,%rdi
  4018f6:	e8 dd 04 00 00       	callq  401dd8 <_ZN4dim3C1Ejjj>
  4018fb:	48 8b 55 e0          	mov    -0x20(%rbp),%rdx
  4018ff:	8b 4d e8             	mov    -0x18(%rbp),%ecx
  401902:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  401906:	8b 45 d8             	mov    -0x28(%rbp),%eax
  401909:	41 b9 00 00 00 00    	mov    $0x0,%r9d
  40190f:	41 b8 00 00 00 00    	mov    $0x0,%r8d
  401915:	48 89 f7             	mov    %rsi,%rdi
  401918:	89 c6                	mov    %eax,%esi
  40191a:	e8 21 f8 ff ff       	callq  401140 <__cudaPushCallConfiguration@plt>
  40191f:	85 c0                	test   %eax,%eax
  401921:	75 17                	jne    40193a <_Z9vectorAddPfS_S_i+0x153>
  401923:	48 8b 55 c8          	mov    -0x38(%rbp),%rdx
  401927:	48 8b 75 c0          	mov    -0x40(%rbp),%rsi
  40192b:	48 8b 45 b8          	mov    -0x48(%rbp),%rax
  40192f:	8b 4d 94             	mov    -0x6c(%rbp),%ecx
  401932:	48 89 c7             	mov    %rax,%rdi
  401935:	e8 b5 06 00 00       	callq  401fef <_Z15vectorAddKernelPfS_S_i>
  40193a:	8b 45 94             	mov    -0x6c(%rbp),%eax
  40193d:	48 98                	cltq   
  40193f:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  401946:	00 
  401947:	48 8b 75 c8          	mov    -0x38(%rbp),%rsi
  40194b:	48 8b 45 98          	mov    -0x68(%rbp),%rax
  40194f:	b9 02 00 00 00       	mov    $0x2,%ecx
  401954:	48 89 c7             	mov    %rax,%rdi
  401957:	e8 54 f7 ff ff       	callq  4010b0 <cudaMemcpy@plt>
  40195c:	48 8b 45 b8          	mov    -0x48(%rbp),%rax
  401960:	48 89 c7             	mov    %rax,%rdi
  401963:	e8 28 f8 ff ff       	callq  401190 <cudaFree@plt>
  401968:	48 8b 45 c0          	mov    -0x40(%rbp),%rax
  40196c:	48 89 c7             	mov    %rax,%rdi
  40196f:	e8 1c f8 ff ff       	callq  401190 <cudaFree@plt>
  401974:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
  401978:	48 89 c7             	mov    %rax,%rdi
  40197b:	e8 10 f8 ff ff       	callq  401190 <cudaFree@plt>
  401980:	90                   	nop
  401981:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  401985:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  40198c:	00 00 
  40198e:	74 0f                	je     40199f <_Z9vectorAddPfS_S_i+0x1b8>
  401990:	eb 08                	jmp    40199a <_Z9vectorAddPfS_S_i+0x1b3>
  401992:	48 89 c7             	mov    %rax,%rdi
  401995:	e8 d6 f7 ff ff       	callq  401170 <_Unwind_Resume@plt>
  40199a:	e8 51 f7 ff ff       	callq  4010f0 <__stack_chk_fail@plt>
  40199f:	c9                   	leaveq 
  4019a0:	c3                   	retq   

00000000004019a1 <_Z9vectorSubPfS_S_i>:
  4019a1:	55                   	push   %rbp
  4019a2:	48 89 e5             	mov    %rsp,%rbp
  4019a5:	48 83 ec 70          	sub    $0x70,%rsp
  4019a9:	48 89 7d a8          	mov    %rdi,-0x58(%rbp)
  4019ad:	48 89 75 a0          	mov    %rsi,-0x60(%rbp)
  4019b1:	48 89 55 98          	mov    %rdx,-0x68(%rbp)
  4019b5:	89 4d 94             	mov    %ecx,-0x6c(%rbp)
  4019b8:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4019bf:	00 00 
  4019c1:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  4019c5:	31 c0                	xor    %eax,%eax
  4019c7:	8b 45 94             	mov    -0x6c(%rbp),%eax
  4019ca:	48 98                	cltq   
  4019cc:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  4019d3:	00 
  4019d4:	48 8d 45 b8          	lea    -0x48(%rbp),%rax
  4019d8:	48 89 d6             	mov    %rdx,%rsi
  4019db:	48 89 c7             	mov    %rax,%rdi
  4019de:	e8 5d f6 ff ff       	callq  401040 <cudaMalloc@plt>
  4019e3:	8b 45 94             	mov    -0x6c(%rbp),%eax
  4019e6:	48 98                	cltq   
  4019e8:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  4019ef:	00 
  4019f0:	48 8b 45 b8          	mov    -0x48(%rbp),%rax
  4019f4:	48 8b 75 a8          	mov    -0x58(%rbp),%rsi
  4019f8:	b9 01 00 00 00       	mov    $0x1,%ecx
  4019fd:	48 89 c7             	mov    %rax,%rdi
  401a00:	e8 ab f6 ff ff       	callq  4010b0 <cudaMemcpy@plt>
  401a05:	8b 45 94             	mov    -0x6c(%rbp),%eax
  401a08:	48 98                	cltq   
  401a0a:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  401a11:	00 
  401a12:	48 8d 45 c0          	lea    -0x40(%rbp),%rax
  401a16:	48 89 d6             	mov    %rdx,%rsi
  401a19:	48 89 c7             	mov    %rax,%rdi
  401a1c:	e8 1f f6 ff ff       	callq  401040 <cudaMalloc@plt>
  401a21:	8b 45 94             	mov    -0x6c(%rbp),%eax
  401a24:	48 98                	cltq   
  401a26:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  401a2d:	00 
  401a2e:	48 8b 45 c0          	mov    -0x40(%rbp),%rax
  401a32:	48 8b 75 a0          	mov    -0x60(%rbp),%rsi
  401a36:	b9 01 00 00 00       	mov    $0x1,%ecx
  401a3b:	48 89 c7             	mov    %rax,%rdi
  401a3e:	e8 6d f6 ff ff       	callq  4010b0 <cudaMemcpy@plt>
  401a43:	8b 45 94             	mov    -0x6c(%rbp),%eax
  401a46:	48 98                	cltq   
  401a48:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  401a4f:	00 
  401a50:	48 8d 45 c8          	lea    -0x38(%rbp),%rax
  401a54:	48 89 d6             	mov    %rdx,%rsi
  401a57:	48 89 c7             	mov    %rax,%rdi
  401a5a:	e8 e1 f5 ff ff       	callq  401040 <cudaMalloc@plt>
  401a5f:	8b 45 94             	mov    -0x6c(%rbp),%eax
  401a62:	48 98                	cltq   
  401a64:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  401a6b:	00 
  401a6c:	48 8b 45 c0          	mov    -0x40(%rbp),%rax
  401a70:	48 8b 75 a0          	mov    -0x60(%rbp),%rsi
  401a74:	b9 01 00 00 00       	mov    $0x1,%ecx
  401a79:	48 89 c7             	mov    %rax,%rdi
  401a7c:	e8 2f f6 ff ff       	callq  4010b0 <cudaMemcpy@plt>
  401a81:	8b 75 94             	mov    -0x6c(%rbp),%esi
  401a84:	48 8d 45 e0          	lea    -0x20(%rbp),%rax
  401a88:	b9 01 00 00 00       	mov    $0x1,%ecx
  401a8d:	ba 01 00 00 00       	mov    $0x1,%edx
  401a92:	48 89 c7             	mov    %rax,%rdi
  401a95:	e8 3e 03 00 00       	callq  401dd8 <_ZN4dim3C1Ejjj>
  401a9a:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  401a9e:	b9 01 00 00 00       	mov    $0x1,%ecx
  401aa3:	ba 01 00 00 00       	mov    $0x1,%edx
  401aa8:	be 01 00 00 00       	mov    $0x1,%esi
  401aad:	48 89 c7             	mov    %rax,%rdi
  401ab0:	e8 23 03 00 00       	callq  401dd8 <_ZN4dim3C1Ejjj>
  401ab5:	48 8b 55 e0          	mov    -0x20(%rbp),%rdx
  401ab9:	8b 4d e8             	mov    -0x18(%rbp),%ecx
  401abc:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  401ac0:	8b 45 d8             	mov    -0x28(%rbp),%eax
  401ac3:	41 b9 00 00 00 00    	mov    $0x0,%r9d
  401ac9:	41 b8 00 00 00 00    	mov    $0x0,%r8d
  401acf:	48 89 f7             	mov    %rsi,%rdi
  401ad2:	89 c6                	mov    %eax,%esi
  401ad4:	e8 67 f6 ff ff       	callq  401140 <__cudaPushCallConfiguration@plt>
  401ad9:	85 c0                	test   %eax,%eax
  401adb:	75 17                	jne    401af4 <_Z9vectorSubPfS_S_i+0x153>
  401add:	48 8b 55 c8          	mov    -0x38(%rbp),%rdx
  401ae1:	48 8b 75 c0          	mov    -0x40(%rbp),%rsi
  401ae5:	48 8b 45 b8          	mov    -0x48(%rbp),%rax
  401ae9:	8b 4d 94             	mov    -0x6c(%rbp),%ecx
  401aec:	48 89 c7             	mov    %rax,%rdi
  401aef:	e8 ac 06 00 00       	callq  4021a0 <_Z15vectorSubKernelPfS_S_i>
  401af4:	8b 45 94             	mov    -0x6c(%rbp),%eax
  401af7:	48 98                	cltq   
  401af9:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  401b00:	00 
  401b01:	48 8b 75 c8          	mov    -0x38(%rbp),%rsi
  401b05:	48 8b 45 98          	mov    -0x68(%rbp),%rax
  401b09:	b9 02 00 00 00       	mov    $0x2,%ecx
  401b0e:	48 89 c7             	mov    %rax,%rdi
  401b11:	e8 9a f5 ff ff       	callq  4010b0 <cudaMemcpy@plt>
  401b16:	48 8b 45 b8          	mov    -0x48(%rbp),%rax
  401b1a:	48 89 c7             	mov    %rax,%rdi
  401b1d:	e8 6e f6 ff ff       	callq  401190 <cudaFree@plt>
  401b22:	48 8b 45 c0          	mov    -0x40(%rbp),%rax
  401b26:	48 89 c7             	mov    %rax,%rdi
  401b29:	e8 62 f6 ff ff       	callq  401190 <cudaFree@plt>
  401b2e:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
  401b32:	48 89 c7             	mov    %rax,%rdi
  401b35:	e8 56 f6 ff ff       	callq  401190 <cudaFree@plt>
  401b3a:	90                   	nop
  401b3b:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  401b3f:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  401b46:	00 00 
  401b48:	74 0f                	je     401b59 <_Z9vectorSubPfS_S_i+0x1b8>
  401b4a:	eb 08                	jmp    401b54 <_Z9vectorSubPfS_S_i+0x1b3>
  401b4c:	48 89 c7             	mov    %rax,%rdi
  401b4f:	e8 1c f6 ff ff       	callq  401170 <_Unwind_Resume@plt>
  401b54:	e8 97 f5 ff ff       	callq  4010f0 <__stack_chk_fail@plt>
  401b59:	c9                   	leaveq 
  401b5a:	c3                   	retq   

0000000000401b5b <_Z9vectorMulPfS_S_i>:
  401b5b:	55                   	push   %rbp
  401b5c:	48 89 e5             	mov    %rsp,%rbp
  401b5f:	48 83 ec 70          	sub    $0x70,%rsp
  401b63:	48 89 7d a8          	mov    %rdi,-0x58(%rbp)
  401b67:	48 89 75 a0          	mov    %rsi,-0x60(%rbp)
  401b6b:	48 89 55 98          	mov    %rdx,-0x68(%rbp)
  401b6f:	89 4d 94             	mov    %ecx,-0x6c(%rbp)
  401b72:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  401b79:	00 00 
  401b7b:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  401b7f:	31 c0                	xor    %eax,%eax
  401b81:	8b 45 94             	mov    -0x6c(%rbp),%eax
  401b84:	48 98                	cltq   
  401b86:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  401b8d:	00 
  401b8e:	48 8d 45 b8          	lea    -0x48(%rbp),%rax
  401b92:	48 89 d6             	mov    %rdx,%rsi
  401b95:	48 89 c7             	mov    %rax,%rdi
  401b98:	e8 a3 f4 ff ff       	callq  401040 <cudaMalloc@plt>
  401b9d:	8b 45 94             	mov    -0x6c(%rbp),%eax
  401ba0:	48 98                	cltq   
  401ba2:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  401ba9:	00 
  401baa:	48 8b 45 b8          	mov    -0x48(%rbp),%rax
  401bae:	48 8b 75 a8          	mov    -0x58(%rbp),%rsi
  401bb2:	b9 01 00 00 00       	mov    $0x1,%ecx
  401bb7:	48 89 c7             	mov    %rax,%rdi
  401bba:	e8 f1 f4 ff ff       	callq  4010b0 <cudaMemcpy@plt>
  401bbf:	8b 45 94             	mov    -0x6c(%rbp),%eax
  401bc2:	48 98                	cltq   
  401bc4:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  401bcb:	00 
  401bcc:	48 8d 45 c0          	lea    -0x40(%rbp),%rax
  401bd0:	48 89 d6             	mov    %rdx,%rsi
  401bd3:	48 89 c7             	mov    %rax,%rdi
  401bd6:	e8 65 f4 ff ff       	callq  401040 <cudaMalloc@plt>
  401bdb:	8b 45 94             	mov    -0x6c(%rbp),%eax
  401bde:	48 98                	cltq   
  401be0:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  401be7:	00 
  401be8:	48 8b 45 c0          	mov    -0x40(%rbp),%rax
  401bec:	48 8b 75 a0          	mov    -0x60(%rbp),%rsi
  401bf0:	b9 01 00 00 00       	mov    $0x1,%ecx
  401bf5:	48 89 c7             	mov    %rax,%rdi
  401bf8:	e8 b3 f4 ff ff       	callq  4010b0 <cudaMemcpy@plt>
  401bfd:	8b 45 94             	mov    -0x6c(%rbp),%eax
  401c00:	48 98                	cltq   
  401c02:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  401c09:	00 
  401c0a:	48 8d 45 c8          	lea    -0x38(%rbp),%rax
  401c0e:	48 89 d6             	mov    %rdx,%rsi
  401c11:	48 89 c7             	mov    %rax,%rdi
  401c14:	e8 27 f4 ff ff       	callq  401040 <cudaMalloc@plt>
  401c19:	8b 45 94             	mov    -0x6c(%rbp),%eax
  401c1c:	48 98                	cltq   
  401c1e:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  401c25:	00 
  401c26:	48 8b 45 c0          	mov    -0x40(%rbp),%rax
  401c2a:	48 8b 75 a0          	mov    -0x60(%rbp),%rsi
  401c2e:	b9 01 00 00 00       	mov    $0x1,%ecx
  401c33:	48 89 c7             	mov    %rax,%rdi
  401c36:	e8 75 f4 ff ff       	callq  4010b0 <cudaMemcpy@plt>
  401c3b:	8b 75 94             	mov    -0x6c(%rbp),%esi
  401c3e:	48 8d 45 e0          	lea    -0x20(%rbp),%rax
  401c42:	b9 01 00 00 00       	mov    $0x1,%ecx
  401c47:	ba 01 00 00 00       	mov    $0x1,%edx
  401c4c:	48 89 c7             	mov    %rax,%rdi
  401c4f:	e8 84 01 00 00       	callq  401dd8 <_ZN4dim3C1Ejjj>
  401c54:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  401c58:	b9 01 00 00 00       	mov    $0x1,%ecx
  401c5d:	ba 01 00 00 00       	mov    $0x1,%edx
  401c62:	be 01 00 00 00       	mov    $0x1,%esi
  401c67:	48 89 c7             	mov    %rax,%rdi
  401c6a:	e8 69 01 00 00       	callq  401dd8 <_ZN4dim3C1Ejjj>
  401c6f:	48 8b 55 e0          	mov    -0x20(%rbp),%rdx
  401c73:	8b 4d e8             	mov    -0x18(%rbp),%ecx
  401c76:	48 8b 75 d0          	mov    -0x30(%rbp),%rsi
  401c7a:	8b 45 d8             	mov    -0x28(%rbp),%eax
  401c7d:	41 b9 00 00 00 00    	mov    $0x0,%r9d
  401c83:	41 b8 00 00 00 00    	mov    $0x0,%r8d
  401c89:	48 89 f7             	mov    %rsi,%rdi
  401c8c:	89 c6                	mov    %eax,%esi
  401c8e:	e8 ad f4 ff ff       	callq  401140 <__cudaPushCallConfiguration@plt>
  401c93:	85 c0                	test   %eax,%eax
  401c95:	75 17                	jne    401cae <_Z9vectorMulPfS_S_i+0x153>
  401c97:	48 8b 55 c8          	mov    -0x38(%rbp),%rdx
  401c9b:	48 8b 75 c0          	mov    -0x40(%rbp),%rsi
  401c9f:	48 8b 45 b8          	mov    -0x48(%rbp),%rax
  401ca3:	8b 4d 94             	mov    -0x6c(%rbp),%ecx
  401ca6:	48 89 c7             	mov    %rax,%rdi
  401ca9:	e8 a3 06 00 00       	callq  402351 <_Z15vectorMulKernelPfS_S_i>
  401cae:	8b 45 94             	mov    -0x6c(%rbp),%eax
  401cb1:	48 98                	cltq   
  401cb3:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
  401cba:	00 
  401cbb:	48 8b 75 c8          	mov    -0x38(%rbp),%rsi
  401cbf:	48 8b 45 98          	mov    -0x68(%rbp),%rax
  401cc3:	b9 02 00 00 00       	mov    $0x2,%ecx
  401cc8:	48 89 c7             	mov    %rax,%rdi
  401ccb:	e8 e0 f3 ff ff       	callq  4010b0 <cudaMemcpy@plt>
  401cd0:	48 8b 45 b8          	mov    -0x48(%rbp),%rax
  401cd4:	48 89 c7             	mov    %rax,%rdi
  401cd7:	e8 b4 f4 ff ff       	callq  401190 <cudaFree@plt>
  401cdc:	48 8b 45 c0          	mov    -0x40(%rbp),%rax
  401ce0:	48 89 c7             	mov    %rax,%rdi
  401ce3:	e8 a8 f4 ff ff       	callq  401190 <cudaFree@plt>
  401ce8:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
  401cec:	48 89 c7             	mov    %rax,%rdi
  401cef:	e8 9c f4 ff ff       	callq  401190 <cudaFree@plt>
  401cf4:	90                   	nop
  401cf5:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  401cf9:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  401d00:	00 00 
  401d02:	74 0f                	je     401d13 <_Z9vectorMulPfS_S_i+0x1b8>
  401d04:	eb 08                	jmp    401d0e <_Z9vectorMulPfS_S_i+0x1b3>
  401d06:	48 89 c7             	mov    %rax,%rdi
  401d09:	e8 62 f4 ff ff       	callq  401170 <_Unwind_Resume@plt>
  401d0e:	e8 dd f3 ff ff       	callq  4010f0 <__stack_chk_fail@plt>
  401d13:	c9                   	leaveq 
  401d14:	c3                   	retq   

0000000000401d15 <_ZL22____nv_dummy_param_refPv>:
  401d15:	55                   	push   %rbp
  401d16:	48 89 e5             	mov    %rsp,%rbp
  401d19:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  401d1d:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  401d21:	48 89 05 10 54 00 00 	mov    %rax,0x5410(%rip)        # 407138 <_ZZL22____nv_dummy_param_refPvE5__ref>
  401d28:	90                   	nop
  401d29:	5d                   	pop    %rbp
  401d2a:	c3                   	retq   

0000000000401d2b <_ZL26__cudaUnregisterBinaryUtilv>:
  401d2b:	55                   	push   %rbp
  401d2c:	48 89 e5             	mov    %rsp,%rbp
  401d2f:	bf 30 71 40 00       	mov    $0x407130,%edi
  401d34:	e8 dc ff ff ff       	callq  401d15 <_ZL22____nv_dummy_param_refPv>
  401d39:	48 8b 05 f0 53 00 00 	mov    0x53f0(%rip),%rax        # 407130 <_ZL20__cudaFatCubinHandle>
  401d40:	48 89 c7             	mov    %rax,%rdi
  401d43:	e8 98 f3 ff ff       	callq  4010e0 <__cudaUnregisterFatBinary@plt>
  401d48:	90                   	nop
  401d49:	5d                   	pop    %rbp
  401d4a:	c3                   	retq   

0000000000401d4b <_ZL32__nv_init_managed_rt_with_modulePPv>:
  401d4b:	55                   	push   %rbp
  401d4c:	48 89 e5             	mov    %rsp,%rbp
  401d4f:	48 83 ec 10          	sub    $0x10,%rsp
  401d53:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  401d57:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  401d5b:	48 89 c7             	mov    %rax,%rdi
  401d5e:	e8 bd f3 ff ff       	callq  401120 <__cudaInitModule@plt>
  401d63:	c9                   	leaveq 
  401d64:	c3                   	retq   

0000000000401d65 <_ZL31__nv_cudaEntityRegisterCallbackPPv>:
  401d65:	55                   	push   %rbp
  401d66:	48 89 e5             	mov    %rsp,%rbp
  401d69:	48 83 ec 08          	sub    $0x8,%rsp
  401d6d:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  401d71:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  401d75:	48 89 05 c4 53 00 00 	mov    %rax,0x53c4(%rip)        # 407140 <_ZZL31__nv_cudaEntityRegisterCallbackPPvE5__ref>
  401d7c:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  401d80:	48 89 c7             	mov    %rax,%rdi
  401d83:	e8 49 fa ff ff       	callq  4017d1 <_ZL37__nv_save_fatbinhandle_for_managed_rtPPv>
  401d88:	90                   	nop
  401d89:	c9                   	leaveq 
  401d8a:	c3                   	retq   

0000000000401d8b <_ZL24__sti____cudaRegisterAllv>:
  401d8b:	55                   	push   %rbp
  401d8c:	48 89 e5             	mov    %rsp,%rbp
  401d8f:	48 83 ec 10          	sub    $0x10,%rsp
  401d93:	bf d8 51 40 00       	mov    $0x4051d8,%edi
  401d98:	e8 23 f3 ff ff       	callq  4010c0 <__cudaRegisterFatBinary@plt>
  401d9d:	48 89 05 8c 53 00 00 	mov    %rax,0x538c(%rip)        # 407130 <_ZL20__cudaFatCubinHandle>
  401da4:	48 c7 45 f8 65 1d 40 	movq   $0x401d65,-0x8(%rbp)
  401dab:	00 
  401dac:	48 8b 15 7d 53 00 00 	mov    0x537d(%rip),%rdx        # 407130 <_ZL20__cudaFatCubinHandle>
  401db3:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  401db7:	48 89 d7             	mov    %rdx,%rdi
  401dba:	ff d0                	callq  *%rax
  401dbc:	48 8b 05 6d 53 00 00 	mov    0x536d(%rip),%rax        # 407130 <_ZL20__cudaFatCubinHandle>
  401dc3:	48 89 c7             	mov    %rax,%rdi
  401dc6:	e8 d5 f2 ff ff       	callq  4010a0 <__cudaRegisterFatBinaryEnd@plt>
  401dcb:	bf 2b 1d 40 00       	mov    $0x401d2b,%edi
  401dd0:	e8 9b 07 00 00       	callq  402570 <atexit>
  401dd5:	90                   	nop
  401dd6:	c9                   	leaveq 
  401dd7:	c3                   	retq   

0000000000401dd8 <_ZN4dim3C1Ejjj>:
  401dd8:	55                   	push   %rbp
  401dd9:	48 89 e5             	mov    %rsp,%rbp
  401ddc:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  401de0:	89 75 f4             	mov    %esi,-0xc(%rbp)
  401de3:	89 55 f0             	mov    %edx,-0x10(%rbp)
  401de6:	89 4d ec             	mov    %ecx,-0x14(%rbp)
  401de9:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  401ded:	8b 55 f4             	mov    -0xc(%rbp),%edx
  401df0:	89 10                	mov    %edx,(%rax)
  401df2:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  401df6:	8b 55 f0             	mov    -0x10(%rbp),%edx
  401df9:	89 50 04             	mov    %edx,0x4(%rax)
  401dfc:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  401e00:	8b 55 ec             	mov    -0x14(%rbp),%edx
  401e03:	89 50 08             	mov    %edx,0x8(%rax)
  401e06:	90                   	nop
  401e07:	5d                   	pop    %rbp
  401e08:	c3                   	retq   

0000000000401e09 <_ZL37__nv_save_fatbinhandle_for_managed_rtPPv>:
  401e09:	55                   	push   %rbp
  401e0a:	48 89 e5             	mov    %rsp,%rbp
  401e0d:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  401e11:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  401e15:	48 89 05 34 53 00 00 	mov    %rax,0x5334(%rip)        # 407150 <_ZL32__nv_fatbinhandle_for_managed_rt>
  401e1c:	90                   	nop
  401e1d:	5d                   	pop    %rbp
  401e1e:	c3                   	retq   

0000000000401e1f <_ZL22____nv_dummy_param_refPv>:
  401e1f:	55                   	push   %rbp
  401e20:	48 89 e5             	mov    %rsp,%rbp
  401e23:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  401e27:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  401e2b:	48 89 05 2e 53 00 00 	mov    %rax,0x532e(%rip)        # 407160 <_ZZL22____nv_dummy_param_refPvE5__ref>
  401e32:	90                   	nop
  401e33:	5d                   	pop    %rbp
  401e34:	c3                   	retq   

0000000000401e35 <_ZL26__cudaUnregisterBinaryUtilv>:
  401e35:	55                   	push   %rbp
  401e36:	48 89 e5             	mov    %rsp,%rbp
  401e39:	bf 58 71 40 00       	mov    $0x407158,%edi
  401e3e:	e8 dc ff ff ff       	callq  401e1f <_ZL22____nv_dummy_param_refPv>
  401e43:	48 8b 05 0e 53 00 00 	mov    0x530e(%rip),%rax        # 407158 <_ZL20__cudaFatCubinHandle>
  401e4a:	48 89 c7             	mov    %rax,%rdi
  401e4d:	e8 8e f2 ff ff       	callq  4010e0 <__cudaUnregisterFatBinary@plt>
  401e52:	90                   	nop
  401e53:	5d                   	pop    %rbp
  401e54:	c3                   	retq   

0000000000401e55 <_ZL32__nv_init_managed_rt_with_modulePPv>:
  401e55:	55                   	push   %rbp
  401e56:	48 89 e5             	mov    %rsp,%rbp
  401e59:	48 83 ec 10          	sub    $0x10,%rsp
  401e5d:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  401e61:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  401e65:	48 89 c7             	mov    %rax,%rdi
  401e68:	e8 b3 f2 ff ff       	callq  401120 <__cudaInitModule@plt>
  401e6d:	c9                   	leaveq 
  401e6e:	c3                   	retq   

0000000000401e6f <_Z40__device_stub__Z15vectorAddKernelPfS_S_iPfS_S_i>:
  401e6f:	55                   	push   %rbp
  401e70:	48 89 e5             	mov    %rsp,%rbp
  401e73:	48 81 ec 90 00 00 00 	sub    $0x90,%rsp
  401e7a:	48 89 7d 88          	mov    %rdi,-0x78(%rbp)
  401e7e:	48 89 75 80          	mov    %rsi,-0x80(%rbp)
  401e82:	48 89 95 78 ff ff ff 	mov    %rdx,-0x88(%rbp)
  401e89:	89 8d 74 ff ff ff    	mov    %ecx,-0x8c(%rbp)
  401e8f:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  401e96:	00 00 
  401e98:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  401e9c:	31 c0                	xor    %eax,%eax
  401e9e:	c7 45 9c 00 00 00 00 	movl   $0x0,-0x64(%rbp)
  401ea5:	8b 45 9c             	mov    -0x64(%rbp),%eax
  401ea8:	48 98                	cltq   
  401eaa:	48 8d 55 88          	lea    -0x78(%rbp),%rdx
  401eae:	48 89 54 c5 d0       	mov    %rdx,-0x30(%rbp,%rax,8)
  401eb3:	83 45 9c 01          	addl   $0x1,-0x64(%rbp)
  401eb7:	8b 45 9c             	mov    -0x64(%rbp),%eax
  401eba:	48 98                	cltq   
  401ebc:	48 8d 55 80          	lea    -0x80(%rbp),%rdx
  401ec0:	48 89 54 c5 d0       	mov    %rdx,-0x30(%rbp,%rax,8)
  401ec5:	83 45 9c 01          	addl   $0x1,-0x64(%rbp)
  401ec9:	8b 45 9c             	mov    -0x64(%rbp),%eax
  401ecc:	48 98                	cltq   
  401ece:	48 8d 95 78 ff ff ff 	lea    -0x88(%rbp),%rdx
  401ed5:	48 89 54 c5 d0       	mov    %rdx,-0x30(%rbp,%rax,8)
  401eda:	83 45 9c 01          	addl   $0x1,-0x64(%rbp)
  401ede:	8b 45 9c             	mov    -0x64(%rbp),%eax
  401ee1:	48 98                	cltq   
  401ee3:	48 8d 95 74 ff ff ff 	lea    -0x8c(%rbp),%rdx
  401eea:	48 89 54 c5 d0       	mov    %rdx,-0x30(%rbp,%rax,8)
  401eef:	83 45 9c 01          	addl   $0x1,-0x64(%rbp)
  401ef3:	48 c7 05 6a 52 00 00 	movq   $0x401fef,0x526a(%rip)        # 407168 <_ZZ40__device_stub__Z15vectorAddKernelPfS_S_iPfS_S_iE3__f>
  401efa:	ef 1f 40 00 
  401efe:	48 8d 45 b0          	lea    -0x50(%rbp),%rax
  401f02:	b9 01 00 00 00       	mov    $0x1,%ecx
  401f07:	ba 01 00 00 00       	mov    $0x1,%edx
  401f0c:	be 01 00 00 00       	mov    $0x1,%esi
  401f11:	48 89 c7             	mov    %rax,%rdi
  401f14:	e8 bf fe ff ff       	callq  401dd8 <_ZN4dim3C1Ejjj>
  401f19:	48 8d 45 c0          	lea    -0x40(%rbp),%rax
  401f1d:	b9 01 00 00 00       	mov    $0x1,%ecx
  401f22:	ba 01 00 00 00       	mov    $0x1,%edx
  401f27:	be 01 00 00 00       	mov    $0x1,%esi
  401f2c:	48 89 c7             	mov    %rax,%rdi
  401f2f:	e8 a4 fe ff ff       	callq  401dd8 <_ZN4dim3C1Ejjj>
  401f34:	48 8d 4d a8          	lea    -0x58(%rbp),%rcx
  401f38:	48 8d 55 a0          	lea    -0x60(%rbp),%rdx
  401f3c:	48 8d 75 c0          	lea    -0x40(%rbp),%rsi
  401f40:	48 8d 45 b0          	lea    -0x50(%rbp),%rax
  401f44:	48 89 c7             	mov    %rax,%rdi
  401f47:	e8 04 f2 ff ff       	callq  401150 <__cudaPopCallConfiguration@plt>
  401f4c:	85 c0                	test   %eax,%eax
  401f4e:	0f 95 c0             	setne  %al
  401f51:	84 c0                	test   %al,%al
  401f53:	0f 85 80 00 00 00    	jne    401fd9 <_Z40__device_stub__Z15vectorAddKernelPfS_S_iPfS_S_i+0x16a>
  401f59:	83 7d 9c 00          	cmpl   $0x0,-0x64(%rbp)
  401f5d:	75 40                	jne    401f9f <_Z40__device_stub__Z15vectorAddKernelPfS_S_iPfS_S_i+0x130>
  401f5f:	48 8b 7d a8          	mov    -0x58(%rbp),%rdi
  401f63:	48 8b 75 a0          	mov    -0x60(%rbp),%rsi
  401f67:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  401f6b:	8b 55 9c             	mov    -0x64(%rbp),%edx
  401f6e:	48 63 d2             	movslq %edx,%rdx
  401f71:	48 c1 e2 03          	shl    $0x3,%rdx
  401f75:	4c 8d 0c 10          	lea    (%rax,%rdx,1),%r9
  401f79:	48 8b 4d c0          	mov    -0x40(%rbp),%rcx
  401f7d:	44 8b 45 c8          	mov    -0x38(%rbp),%r8d
  401f81:	48 8b 55 b0          	mov    -0x50(%rbp),%rdx
  401f85:	8b 45 b8             	mov    -0x48(%rbp),%eax
  401f88:	57                   	push   %rdi
  401f89:	56                   	push   %rsi
  401f8a:	48 89 d6             	mov    %rdx,%rsi
  401f8d:	89 c2                	mov    %eax,%edx
  401f8f:	bf ef 1f 40 00       	mov    $0x401fef,%edi
  401f94:	e8 f5 04 00 00       	callq  40248e <_Z16cudaLaunchKernelIcE9cudaErrorPKT_4dim3S4_PPvmP11CUstream_st>
  401f99:	48 83 c4 10          	add    $0x10,%rsp
  401f9d:	eb 3a                	jmp    401fd9 <_Z40__device_stub__Z15vectorAddKernelPfS_S_iPfS_S_i+0x16a>
  401f9f:	48 8b 7d a8          	mov    -0x58(%rbp),%rdi
  401fa3:	48 8b 75 a0          	mov    -0x60(%rbp),%rsi
  401fa7:	4c 8d 4d d0          	lea    -0x30(%rbp),%r9
  401fab:	48 8b 4d c0          	mov    -0x40(%rbp),%rcx
  401faf:	44 8b 45 c8          	mov    -0x38(%rbp),%r8d
  401fb3:	48 8b 55 b0          	mov    -0x50(%rbp),%rdx
  401fb7:	8b 45 b8             	mov    -0x48(%rbp),%eax
  401fba:	57                   	push   %rdi
  401fbb:	56                   	push   %rsi
  401fbc:	48 89 d6             	mov    %rdx,%rsi
  401fbf:	89 c2                	mov    %eax,%edx
  401fc1:	bf ef 1f 40 00       	mov    $0x401fef,%edi
  401fc6:	e8 c3 04 00 00       	callq  40248e <_Z16cudaLaunchKernelIcE9cudaErrorPKT_4dim3S4_PPvmP11CUstream_st>
  401fcb:	48 83 c4 10          	add    $0x10,%rsp
  401fcf:	eb 08                	jmp    401fd9 <_Z40__device_stub__Z15vectorAddKernelPfS_S_iPfS_S_i+0x16a>
  401fd1:	48 89 c7             	mov    %rax,%rdi
  401fd4:	e8 97 f1 ff ff       	callq  401170 <_Unwind_Resume@plt>
  401fd9:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  401fdd:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  401fe4:	00 00 
  401fe6:	74 05                	je     401fed <_Z40__device_stub__Z15vectorAddKernelPfS_S_iPfS_S_i+0x17e>
  401fe8:	e8 03 f1 ff ff       	callq  4010f0 <__stack_chk_fail@plt>
  401fed:	c9                   	leaveq 
  401fee:	c3                   	retq   

0000000000401fef <_Z15vectorAddKernelPfS_S_i>:
  401fef:	55                   	push   %rbp
  401ff0:	48 89 e5             	mov    %rsp,%rbp
  401ff3:	48 83 ec 20          	sub    $0x20,%rsp
  401ff7:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  401ffb:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  401fff:	48 89 55 e8          	mov    %rdx,-0x18(%rbp)
  402003:	89 4d e4             	mov    %ecx,-0x1c(%rbp)
  402006:	8b 4d e4             	mov    -0x1c(%rbp),%ecx
  402009:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
  40200d:	48 8b 75 f0          	mov    -0x10(%rbp),%rsi
  402011:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  402015:	48 89 c7             	mov    %rax,%rdi
  402018:	e8 52 fe ff ff       	callq  401e6f <_Z40__device_stub__Z15vectorAddKernelPfS_S_iPfS_S_i>
  40201d:	90                   	nop
  40201e:	c9                   	leaveq 
  40201f:	c3                   	retq   

0000000000402020 <_Z40__device_stub__Z15vectorSubKernelPfS_S_iPfS_S_i>:
  402020:	55                   	push   %rbp
  402021:	48 89 e5             	mov    %rsp,%rbp
  402024:	48 81 ec 90 00 00 00 	sub    $0x90,%rsp
  40202b:	48 89 7d 88          	mov    %rdi,-0x78(%rbp)
  40202f:	48 89 75 80          	mov    %rsi,-0x80(%rbp)
  402033:	48 89 95 78 ff ff ff 	mov    %rdx,-0x88(%rbp)
  40203a:	89 8d 74 ff ff ff    	mov    %ecx,-0x8c(%rbp)
  402040:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  402047:	00 00 
  402049:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  40204d:	31 c0                	xor    %eax,%eax
  40204f:	c7 45 9c 00 00 00 00 	movl   $0x0,-0x64(%rbp)
  402056:	8b 45 9c             	mov    -0x64(%rbp),%eax
  402059:	48 98                	cltq   
  40205b:	48 8d 55 88          	lea    -0x78(%rbp),%rdx
  40205f:	48 89 54 c5 d0       	mov    %rdx,-0x30(%rbp,%rax,8)
  402064:	83 45 9c 01          	addl   $0x1,-0x64(%rbp)
  402068:	8b 45 9c             	mov    -0x64(%rbp),%eax
  40206b:	48 98                	cltq   
  40206d:	48 8d 55 80          	lea    -0x80(%rbp),%rdx
  402071:	48 89 54 c5 d0       	mov    %rdx,-0x30(%rbp,%rax,8)
  402076:	83 45 9c 01          	addl   $0x1,-0x64(%rbp)
  40207a:	8b 45 9c             	mov    -0x64(%rbp),%eax
  40207d:	48 98                	cltq   
  40207f:	48 8d 95 78 ff ff ff 	lea    -0x88(%rbp),%rdx
  402086:	48 89 54 c5 d0       	mov    %rdx,-0x30(%rbp,%rax,8)
  40208b:	83 45 9c 01          	addl   $0x1,-0x64(%rbp)
  40208f:	8b 45 9c             	mov    -0x64(%rbp),%eax
  402092:	48 98                	cltq   
  402094:	48 8d 95 74 ff ff ff 	lea    -0x8c(%rbp),%rdx
  40209b:	48 89 54 c5 d0       	mov    %rdx,-0x30(%rbp,%rax,8)
  4020a0:	83 45 9c 01          	addl   $0x1,-0x64(%rbp)
  4020a4:	48 c7 05 c1 50 00 00 	movq   $0x4021a0,0x50c1(%rip)        # 407170 <_ZZ40__device_stub__Z15vectorSubKernelPfS_S_iPfS_S_iE3__f>
  4020ab:	a0 21 40 00 
  4020af:	48 8d 45 b0          	lea    -0x50(%rbp),%rax
  4020b3:	b9 01 00 00 00       	mov    $0x1,%ecx
  4020b8:	ba 01 00 00 00       	mov    $0x1,%edx
  4020bd:	be 01 00 00 00       	mov    $0x1,%esi
  4020c2:	48 89 c7             	mov    %rax,%rdi
  4020c5:	e8 0e fd ff ff       	callq  401dd8 <_ZN4dim3C1Ejjj>
  4020ca:	48 8d 45 c0          	lea    -0x40(%rbp),%rax
  4020ce:	b9 01 00 00 00       	mov    $0x1,%ecx
  4020d3:	ba 01 00 00 00       	mov    $0x1,%edx
  4020d8:	be 01 00 00 00       	mov    $0x1,%esi
  4020dd:	48 89 c7             	mov    %rax,%rdi
  4020e0:	e8 f3 fc ff ff       	callq  401dd8 <_ZN4dim3C1Ejjj>
  4020e5:	48 8d 4d a8          	lea    -0x58(%rbp),%rcx
  4020e9:	48 8d 55 a0          	lea    -0x60(%rbp),%rdx
  4020ed:	48 8d 75 c0          	lea    -0x40(%rbp),%rsi
  4020f1:	48 8d 45 b0          	lea    -0x50(%rbp),%rax
  4020f5:	48 89 c7             	mov    %rax,%rdi
  4020f8:	e8 53 f0 ff ff       	callq  401150 <__cudaPopCallConfiguration@plt>
  4020fd:	85 c0                	test   %eax,%eax
  4020ff:	0f 95 c0             	setne  %al
  402102:	84 c0                	test   %al,%al
  402104:	0f 85 80 00 00 00    	jne    40218a <_Z40__device_stub__Z15vectorSubKernelPfS_S_iPfS_S_i+0x16a>
  40210a:	83 7d 9c 00          	cmpl   $0x0,-0x64(%rbp)
  40210e:	75 40                	jne    402150 <_Z40__device_stub__Z15vectorSubKernelPfS_S_iPfS_S_i+0x130>
  402110:	48 8b 7d a8          	mov    -0x58(%rbp),%rdi
  402114:	48 8b 75 a0          	mov    -0x60(%rbp),%rsi
  402118:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  40211c:	8b 55 9c             	mov    -0x64(%rbp),%edx
  40211f:	48 63 d2             	movslq %edx,%rdx
  402122:	48 c1 e2 03          	shl    $0x3,%rdx
  402126:	4c 8d 0c 10          	lea    (%rax,%rdx,1),%r9
  40212a:	48 8b 4d c0          	mov    -0x40(%rbp),%rcx
  40212e:	44 8b 45 c8          	mov    -0x38(%rbp),%r8d
  402132:	48 8b 55 b0          	mov    -0x50(%rbp),%rdx
  402136:	8b 45 b8             	mov    -0x48(%rbp),%eax
  402139:	57                   	push   %rdi
  40213a:	56                   	push   %rsi
  40213b:	48 89 d6             	mov    %rdx,%rsi
  40213e:	89 c2                	mov    %eax,%edx
  402140:	bf a0 21 40 00       	mov    $0x4021a0,%edi
  402145:	e8 44 03 00 00       	callq  40248e <_Z16cudaLaunchKernelIcE9cudaErrorPKT_4dim3S4_PPvmP11CUstream_st>
  40214a:	48 83 c4 10          	add    $0x10,%rsp
  40214e:	eb 3a                	jmp    40218a <_Z40__device_stub__Z15vectorSubKernelPfS_S_iPfS_S_i+0x16a>
  402150:	48 8b 7d a8          	mov    -0x58(%rbp),%rdi
  402154:	48 8b 75 a0          	mov    -0x60(%rbp),%rsi
  402158:	4c 8d 4d d0          	lea    -0x30(%rbp),%r9
  40215c:	48 8b 4d c0          	mov    -0x40(%rbp),%rcx
  402160:	44 8b 45 c8          	mov    -0x38(%rbp),%r8d
  402164:	48 8b 55 b0          	mov    -0x50(%rbp),%rdx
  402168:	8b 45 b8             	mov    -0x48(%rbp),%eax
  40216b:	57                   	push   %rdi
  40216c:	56                   	push   %rsi
  40216d:	48 89 d6             	mov    %rdx,%rsi
  402170:	89 c2                	mov    %eax,%edx
  402172:	bf a0 21 40 00       	mov    $0x4021a0,%edi
  402177:	e8 12 03 00 00       	callq  40248e <_Z16cudaLaunchKernelIcE9cudaErrorPKT_4dim3S4_PPvmP11CUstream_st>
  40217c:	48 83 c4 10          	add    $0x10,%rsp
  402180:	eb 08                	jmp    40218a <_Z40__device_stub__Z15vectorSubKernelPfS_S_iPfS_S_i+0x16a>
  402182:	48 89 c7             	mov    %rax,%rdi
  402185:	e8 e6 ef ff ff       	callq  401170 <_Unwind_Resume@plt>
  40218a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40218e:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  402195:	00 00 
  402197:	74 05                	je     40219e <_Z40__device_stub__Z15vectorSubKernelPfS_S_iPfS_S_i+0x17e>
  402199:	e8 52 ef ff ff       	callq  4010f0 <__stack_chk_fail@plt>
  40219e:	c9                   	leaveq 
  40219f:	c3                   	retq   

00000000004021a0 <_Z15vectorSubKernelPfS_S_i>:
  4021a0:	55                   	push   %rbp
  4021a1:	48 89 e5             	mov    %rsp,%rbp
  4021a4:	48 83 ec 20          	sub    $0x20,%rsp
  4021a8:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  4021ac:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  4021b0:	48 89 55 e8          	mov    %rdx,-0x18(%rbp)
  4021b4:	89 4d e4             	mov    %ecx,-0x1c(%rbp)
  4021b7:	8b 4d e4             	mov    -0x1c(%rbp),%ecx
  4021ba:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
  4021be:	48 8b 75 f0          	mov    -0x10(%rbp),%rsi
  4021c2:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4021c6:	48 89 c7             	mov    %rax,%rdi
  4021c9:	e8 52 fe ff ff       	callq  402020 <_Z40__device_stub__Z15vectorSubKernelPfS_S_iPfS_S_i>
  4021ce:	90                   	nop
  4021cf:	c9                   	leaveq 
  4021d0:	c3                   	retq   

00000000004021d1 <_Z40__device_stub__Z15vectorMulKernelPfS_S_iPfS_S_i>:
  4021d1:	55                   	push   %rbp
  4021d2:	48 89 e5             	mov    %rsp,%rbp
  4021d5:	48 81 ec 90 00 00 00 	sub    $0x90,%rsp
  4021dc:	48 89 7d 88          	mov    %rdi,-0x78(%rbp)
  4021e0:	48 89 75 80          	mov    %rsi,-0x80(%rbp)
  4021e4:	48 89 95 78 ff ff ff 	mov    %rdx,-0x88(%rbp)
  4021eb:	89 8d 74 ff ff ff    	mov    %ecx,-0x8c(%rbp)
  4021f1:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
  4021f8:	00 00 
  4021fa:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
  4021fe:	31 c0                	xor    %eax,%eax
  402200:	c7 45 9c 00 00 00 00 	movl   $0x0,-0x64(%rbp)
  402207:	8b 45 9c             	mov    -0x64(%rbp),%eax
  40220a:	48 98                	cltq   
  40220c:	48 8d 55 88          	lea    -0x78(%rbp),%rdx
  402210:	48 89 54 c5 d0       	mov    %rdx,-0x30(%rbp,%rax,8)
  402215:	83 45 9c 01          	addl   $0x1,-0x64(%rbp)
  402219:	8b 45 9c             	mov    -0x64(%rbp),%eax
  40221c:	48 98                	cltq   
  40221e:	48 8d 55 80          	lea    -0x80(%rbp),%rdx
  402222:	48 89 54 c5 d0       	mov    %rdx,-0x30(%rbp,%rax,8)
  402227:	83 45 9c 01          	addl   $0x1,-0x64(%rbp)
  40222b:	8b 45 9c             	mov    -0x64(%rbp),%eax
  40222e:	48 98                	cltq   
  402230:	48 8d 95 78 ff ff ff 	lea    -0x88(%rbp),%rdx
  402237:	48 89 54 c5 d0       	mov    %rdx,-0x30(%rbp,%rax,8)
  40223c:	83 45 9c 01          	addl   $0x1,-0x64(%rbp)
  402240:	8b 45 9c             	mov    -0x64(%rbp),%eax
  402243:	48 98                	cltq   
  402245:	48 8d 95 74 ff ff ff 	lea    -0x8c(%rbp),%rdx
  40224c:	48 89 54 c5 d0       	mov    %rdx,-0x30(%rbp,%rax,8)
  402251:	83 45 9c 01          	addl   $0x1,-0x64(%rbp)
  402255:	48 c7 05 18 4f 00 00 	movq   $0x402351,0x4f18(%rip)        # 407178 <_ZZ40__device_stub__Z15vectorMulKernelPfS_S_iPfS_S_iE3__f>
  40225c:	51 23 40 00 
  402260:	48 8d 45 b0          	lea    -0x50(%rbp),%rax
  402264:	b9 01 00 00 00       	mov    $0x1,%ecx
  402269:	ba 01 00 00 00       	mov    $0x1,%edx
  40226e:	be 01 00 00 00       	mov    $0x1,%esi
  402273:	48 89 c7             	mov    %rax,%rdi
  402276:	e8 5d fb ff ff       	callq  401dd8 <_ZN4dim3C1Ejjj>
  40227b:	48 8d 45 c0          	lea    -0x40(%rbp),%rax
  40227f:	b9 01 00 00 00       	mov    $0x1,%ecx
  402284:	ba 01 00 00 00       	mov    $0x1,%edx
  402289:	be 01 00 00 00       	mov    $0x1,%esi
  40228e:	48 89 c7             	mov    %rax,%rdi
  402291:	e8 42 fb ff ff       	callq  401dd8 <_ZN4dim3C1Ejjj>
  402296:	48 8d 4d a8          	lea    -0x58(%rbp),%rcx
  40229a:	48 8d 55 a0          	lea    -0x60(%rbp),%rdx
  40229e:	48 8d 75 c0          	lea    -0x40(%rbp),%rsi
  4022a2:	48 8d 45 b0          	lea    -0x50(%rbp),%rax
  4022a6:	48 89 c7             	mov    %rax,%rdi
  4022a9:	e8 a2 ee ff ff       	callq  401150 <__cudaPopCallConfiguration@plt>
  4022ae:	85 c0                	test   %eax,%eax
  4022b0:	0f 95 c0             	setne  %al
  4022b3:	84 c0                	test   %al,%al
  4022b5:	0f 85 80 00 00 00    	jne    40233b <_Z40__device_stub__Z15vectorMulKernelPfS_S_iPfS_S_i+0x16a>
  4022bb:	83 7d 9c 00          	cmpl   $0x0,-0x64(%rbp)
  4022bf:	75 40                	jne    402301 <_Z40__device_stub__Z15vectorMulKernelPfS_S_iPfS_S_i+0x130>
  4022c1:	48 8b 7d a8          	mov    -0x58(%rbp),%rdi
  4022c5:	48 8b 75 a0          	mov    -0x60(%rbp),%rsi
  4022c9:	48 8d 45 d0          	lea    -0x30(%rbp),%rax
  4022cd:	8b 55 9c             	mov    -0x64(%rbp),%edx
  4022d0:	48 63 d2             	movslq %edx,%rdx
  4022d3:	48 c1 e2 03          	shl    $0x3,%rdx
  4022d7:	4c 8d 0c 10          	lea    (%rax,%rdx,1),%r9
  4022db:	48 8b 4d c0          	mov    -0x40(%rbp),%rcx
  4022df:	44 8b 45 c8          	mov    -0x38(%rbp),%r8d
  4022e3:	48 8b 55 b0          	mov    -0x50(%rbp),%rdx
  4022e7:	8b 45 b8             	mov    -0x48(%rbp),%eax
  4022ea:	57                   	push   %rdi
  4022eb:	56                   	push   %rsi
  4022ec:	48 89 d6             	mov    %rdx,%rsi
  4022ef:	89 c2                	mov    %eax,%edx
  4022f1:	bf 51 23 40 00       	mov    $0x402351,%edi
  4022f6:	e8 93 01 00 00       	callq  40248e <_Z16cudaLaunchKernelIcE9cudaErrorPKT_4dim3S4_PPvmP11CUstream_st>
  4022fb:	48 83 c4 10          	add    $0x10,%rsp
  4022ff:	eb 3a                	jmp    40233b <_Z40__device_stub__Z15vectorMulKernelPfS_S_iPfS_S_i+0x16a>
  402301:	48 8b 7d a8          	mov    -0x58(%rbp),%rdi
  402305:	48 8b 75 a0          	mov    -0x60(%rbp),%rsi
  402309:	4c 8d 4d d0          	lea    -0x30(%rbp),%r9
  40230d:	48 8b 4d c0          	mov    -0x40(%rbp),%rcx
  402311:	44 8b 45 c8          	mov    -0x38(%rbp),%r8d
  402315:	48 8b 55 b0          	mov    -0x50(%rbp),%rdx
  402319:	8b 45 b8             	mov    -0x48(%rbp),%eax
  40231c:	57                   	push   %rdi
  40231d:	56                   	push   %rsi
  40231e:	48 89 d6             	mov    %rdx,%rsi
  402321:	89 c2                	mov    %eax,%edx
  402323:	bf 51 23 40 00       	mov    $0x402351,%edi
  402328:	e8 61 01 00 00       	callq  40248e <_Z16cudaLaunchKernelIcE9cudaErrorPKT_4dim3S4_PPvmP11CUstream_st>
  40232d:	48 83 c4 10          	add    $0x10,%rsp
  402331:	eb 08                	jmp    40233b <_Z40__device_stub__Z15vectorMulKernelPfS_S_iPfS_S_i+0x16a>
  402333:	48 89 c7             	mov    %rax,%rdi
  402336:	e8 35 ee ff ff       	callq  401170 <_Unwind_Resume@plt>
  40233b:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40233f:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  402346:	00 00 
  402348:	74 05                	je     40234f <_Z40__device_stub__Z15vectorMulKernelPfS_S_iPfS_S_i+0x17e>
  40234a:	e8 a1 ed ff ff       	callq  4010f0 <__stack_chk_fail@plt>
  40234f:	c9                   	leaveq 
  402350:	c3                   	retq   

0000000000402351 <_Z15vectorMulKernelPfS_S_i>:
  402351:	55                   	push   %rbp
  402352:	48 89 e5             	mov    %rsp,%rbp
  402355:	48 83 ec 20          	sub    $0x20,%rsp
  402359:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40235d:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
  402361:	48 89 55 e8          	mov    %rdx,-0x18(%rbp)
  402365:	89 4d e4             	mov    %ecx,-0x1c(%rbp)
  402368:	8b 4d e4             	mov    -0x1c(%rbp),%ecx
  40236b:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
  40236f:	48 8b 75 f0          	mov    -0x10(%rbp),%rsi
  402373:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  402377:	48 89 c7             	mov    %rax,%rdi
  40237a:	e8 52 fe ff ff       	callq  4021d1 <_Z40__device_stub__Z15vectorMulKernelPfS_S_iPfS_S_i>
  40237f:	90                   	nop
  402380:	c9                   	leaveq 
  402381:	c3                   	retq   

0000000000402382 <_ZL31__nv_cudaEntityRegisterCallbackPPv>:
  402382:	55                   	push   %rbp
  402383:	48 89 e5             	mov    %rsp,%rbp
  402386:	48 83 ec 10          	sub    $0x10,%rsp
  40238a:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40238e:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  402392:	48 89 05 e7 4d 00 00 	mov    %rax,0x4de7(%rip)        # 407180 <_ZZL31__nv_cudaEntityRegisterCallbackPPvE5__ref>
  402399:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40239d:	48 89 c7             	mov    %rax,%rdi
  4023a0:	e8 64 fa ff ff       	callq  401e09 <_ZL37__nv_save_fatbinhandle_for_managed_rtPPv>
  4023a5:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4023a9:	6a 00                	pushq  $0x0
  4023ab:	6a 00                	pushq  $0x0
  4023ad:	6a 00                	pushq  $0x0
  4023af:	6a 00                	pushq  $0x0
  4023b1:	41 b9 00 00 00 00    	mov    $0x0,%r9d
  4023b7:	41 b8 ff ff ff ff    	mov    $0xffffffff,%r8d
  4023bd:	b9 a8 31 40 00       	mov    $0x4031a8,%ecx
  4023c2:	ba a8 31 40 00       	mov    $0x4031a8,%edx
  4023c7:	be 51 23 40 00       	mov    $0x402351,%esi
  4023cc:	48 89 c7             	mov    %rax,%rdi
  4023cf:	e8 2c ed ff ff       	callq  401100 <__cudaRegisterFunction@plt>
  4023d4:	48 83 c4 20          	add    $0x20,%rsp
  4023d8:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4023dc:	6a 00                	pushq  $0x0
  4023de:	6a 00                	pushq  $0x0
  4023e0:	6a 00                	pushq  $0x0
  4023e2:	6a 00                	pushq  $0x0
  4023e4:	41 b9 00 00 00 00    	mov    $0x0,%r9d
  4023ea:	41 b8 ff ff ff ff    	mov    $0xffffffff,%r8d
  4023f0:	b9 c3 31 40 00       	mov    $0x4031c3,%ecx
  4023f5:	ba c3 31 40 00       	mov    $0x4031c3,%edx
  4023fa:	be a0 21 40 00       	mov    $0x4021a0,%esi
  4023ff:	48 89 c7             	mov    %rax,%rdi
  402402:	e8 f9 ec ff ff       	callq  401100 <__cudaRegisterFunction@plt>
  402407:	48 83 c4 20          	add    $0x20,%rsp
  40240b:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40240f:	6a 00                	pushq  $0x0
  402411:	6a 00                	pushq  $0x0
  402413:	6a 00                	pushq  $0x0
  402415:	6a 00                	pushq  $0x0
  402417:	41 b9 00 00 00 00    	mov    $0x0,%r9d
  40241d:	41 b8 ff ff ff ff    	mov    $0xffffffff,%r8d
  402423:	b9 de 31 40 00       	mov    $0x4031de,%ecx
  402428:	ba de 31 40 00       	mov    $0x4031de,%edx
  40242d:	be ef 1f 40 00       	mov    $0x401fef,%esi
  402432:	48 89 c7             	mov    %rax,%rdi
  402435:	e8 c6 ec ff ff       	callq  401100 <__cudaRegisterFunction@plt>
  40243a:	48 83 c4 20          	add    $0x20,%rsp
  40243e:	90                   	nop
  40243f:	c9                   	leaveq 
  402440:	c3                   	retq   

0000000000402441 <_ZL24__sti____cudaRegisterAllv>:
  402441:	55                   	push   %rbp
  402442:	48 89 e5             	mov    %rsp,%rbp
  402445:	48 83 ec 10          	sub    $0x10,%rsp
  402449:	bf f0 51 40 00       	mov    $0x4051f0,%edi
  40244e:	e8 6d ec ff ff       	callq  4010c0 <__cudaRegisterFatBinary@plt>
  402453:	48 89 05 fe 4c 00 00 	mov    %rax,0x4cfe(%rip)        # 407158 <_ZL20__cudaFatCubinHandle>
  40245a:	48 c7 45 f8 82 23 40 	movq   $0x402382,-0x8(%rbp)
  402461:	00 
  402462:	48 8b 15 ef 4c 00 00 	mov    0x4cef(%rip),%rdx        # 407158 <_ZL20__cudaFatCubinHandle>
  402469:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  40246d:	48 89 d7             	mov    %rdx,%rdi
  402470:	ff d0                	callq  *%rax
  402472:	48 8b 05 df 4c 00 00 	mov    0x4cdf(%rip),%rax        # 407158 <_ZL20__cudaFatCubinHandle>
  402479:	48 89 c7             	mov    %rax,%rdi
  40247c:	e8 1f ec ff ff       	callq  4010a0 <__cudaRegisterFatBinaryEnd@plt>
  402481:	bf 35 1e 40 00       	mov    $0x401e35,%edi
  402486:	e8 e5 00 00 00       	callq  402570 <atexit>
  40248b:	90                   	nop
  40248c:	c9                   	leaveq 
  40248d:	c3                   	retq   

000000000040248e <_Z16cudaLaunchKernelIcE9cudaErrorPKT_4dim3S4_PPvmP11CUstream_st>:
  40248e:	55                   	push   %rbp
  40248f:	48 89 e5             	mov    %rsp,%rbp
  402492:	48 83 ec 30          	sub    $0x30,%rsp
  402496:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
  40249a:	48 89 c8             	mov    %rcx,%rax
  40249d:	44 89 c1             	mov    %r8d,%ecx
  4024a0:	4c 89 4d d0          	mov    %r9,-0x30(%rbp)
  4024a4:	48 89 75 e8          	mov    %rsi,-0x18(%rbp)
  4024a8:	89 55 f0             	mov    %edx,-0x10(%rbp)
  4024ab:	48 89 45 d8          	mov    %rax,-0x28(%rbp)
  4024af:	89 4d e0             	mov    %ecx,-0x20(%rbp)
  4024b2:	4c 8b 45 d0          	mov    -0x30(%rbp),%r8
  4024b6:	48 8b 4d d8          	mov    -0x28(%rbp),%rcx
  4024ba:	8b 7d e0             	mov    -0x20(%rbp),%edi
  4024bd:	48 8b 75 e8          	mov    -0x18(%rbp),%rsi
  4024c1:	8b 55 f0             	mov    -0x10(%rbp),%edx
  4024c4:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
  4024c8:	ff 75 18             	pushq  0x18(%rbp)
  4024cb:	ff 75 10             	pushq  0x10(%rbp)
  4024ce:	4d 89 c1             	mov    %r8,%r9
  4024d1:	41 89 f8             	mov    %edi,%r8d
  4024d4:	48 89 c7             	mov    %rax,%rdi
  4024d7:	e8 84 eb ff ff       	callq  401060 <cudaLaunchKernel@plt>
  4024dc:	48 83 c4 10          	add    $0x10,%rsp
  4024e0:	c9                   	leaveq 
  4024e1:	c3                   	retq   
  4024e2:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4024e9:	00 00 00 
  4024ec:	0f 1f 40 00          	nopl   0x0(%rax)

00000000004024f0 <__libc_csu_init>:
  4024f0:	f3 0f 1e fa          	endbr64 
  4024f4:	41 57                	push   %r15
  4024f6:	4c 8d 3d bb 48 00 00 	lea    0x48bb(%rip),%r15        # 406db8 <__frame_dummy_init_array_entry>
  4024fd:	41 56                	push   %r14
  4024ff:	49 89 d6             	mov    %rdx,%r14
  402502:	41 55                	push   %r13
  402504:	49 89 f5             	mov    %rsi,%r13
  402507:	41 54                	push   %r12
  402509:	41 89 fc             	mov    %edi,%r12d
  40250c:	55                   	push   %rbp
  40250d:	48 8d 2d bc 48 00 00 	lea    0x48bc(%rip),%rbp        # 406dd0 <__do_global_dtors_aux_fini_array_entry>
  402514:	53                   	push   %rbx
  402515:	4c 29 fd             	sub    %r15,%rbp
  402518:	48 83 ec 08          	sub    $0x8,%rsp
  40251c:	e8 df ea ff ff       	callq  401000 <_init>
  402521:	48 c1 fd 03          	sar    $0x3,%rbp
  402525:	74 1f                	je     402546 <__libc_csu_init+0x56>
  402527:	31 db                	xor    %ebx,%ebx
  402529:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  402530:	4c 89 f2             	mov    %r14,%rdx
  402533:	4c 89 ee             	mov    %r13,%rsi
  402536:	44 89 e7             	mov    %r12d,%edi
  402539:	41 ff 14 df          	callq  *(%r15,%rbx,8)
  40253d:	48 83 c3 01          	add    $0x1,%rbx
  402541:	48 39 dd             	cmp    %rbx,%rbp
  402544:	75 ea                	jne    402530 <__libc_csu_init+0x40>
  402546:	48 83 c4 08          	add    $0x8,%rsp
  40254a:	5b                   	pop    %rbx
  40254b:	5d                   	pop    %rbp
  40254c:	41 5c                	pop    %r12
  40254e:	41 5d                	pop    %r13
  402550:	41 5e                	pop    %r14
  402552:	41 5f                	pop    %r15
  402554:	c3                   	retq   
  402555:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  40255c:	00 00 00 00 

0000000000402560 <__libc_csu_fini>:
  402560:	f3 0f 1e fa          	endbr64 
  402564:	c3                   	retq   
  402565:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  40256c:	00 00 00 
  40256f:	90                   	nop

0000000000402570 <atexit>:
  402570:	f3 0f 1e fa          	endbr64 
  402574:	48 8b 15 5d 4b 00 00 	mov    0x4b5d(%rip),%rdx        # 4070d8 <__dso_handle>
  40257b:	31 f6                	xor    %esi,%esi
  40257d:	e9 4e eb ff ff       	jmpq   4010d0 <__cxa_atexit@plt>

Disassembly of section .fini:

0000000000402584 <_fini>:
  402584:	f3 0f 1e fa          	endbr64 
  402588:	48 83 ec 08          	sub    $0x8,%rsp
  40258c:	48 83 c4 08          	add    $0x8,%rsp
  402590:	c3                   	retq   
