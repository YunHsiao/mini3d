.model flat, stdcall

.code
; -------------------------------------------------------------------------
sub_ps PROC, z:DWORD, x:DWORD, y:DWORD
	mov    eax, dword ptr [x]
	mov    ecx, dword ptr [y]
	movaps xmm0, xmmword ptr [eax]
	movaps xmm1, xmmword ptr [ecx]
    
	subps  xmm0, xmm1

	mov    eax, dword ptr [z]
    movaps xmmword ptr [eax], xmm0
    ret
sub_ps ENDP
; -------------------------------------------------------------------------
end